import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple
from datasets import load_dataset

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
    PreTrainedTokenizer
)

from peft import LoraConfig, TaskType

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    DatasetConfig,
    SimplePreferenceCollator,
)

from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    get_reward,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    save_with_accelerate,
)

from open_instruct.utils import (
    ArgumentParserPlus,
    get_wandb_tags,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

from utils import (
    PreferencePromptDatasetProcessor,
    INPUT_IDS_PROMPT_KEY,
    LABEL_KEY,
    PROMPT_KEY,
    COMPLETION_KEY,
    INPUT_IDS_CHOSEN_KEY,
    INPUT_IDS_REJECTED_KEY
)


@dataclass
class Args:
    # required dataset args
    dataset_train_splits: str = None
    """The dataset splits to use for training"""
    dataset_eval_splits: str = None
    """The dataset splits to use for evaluation"""
    dataset_name: str = "mbpp"
    """Name of the dataset"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    save_freq: Optional[int] = None
    """The frequency of save steps"""
    max_test_dataset: Optional[int] = None
    """The max test dataset"""
    
    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "LLM_RM"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""
    eval_dir: Optional[str] = None
    """Where to save the evals"""

    resize_token_embeddings: bool = False
    """Whether to resize the token embeddings to a factor of 8 for utilizing tensor cores better"""


def find_shared_text(chosen_text: str, rejected_text: str):
    """return shared (prompt) text between chosen and rejected text"""
    for i in range(min(len(chosen_text), len(rejected_text))):
        if chosen_text[i] != rejected_text[i]:
            break

    return chosen_text[:i]

def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    max_sampled_texts: int = -1
) -> Tuple[dict, dict]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    total_reward_margin = 0
    total_batches = 0
    table = None

    table = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            query_responses = torch.cat((data[INPUT_IDS_CHOSEN_KEY], data[INPUT_IDS_REJECTED_KEY]), dim=0)
            _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
            chosen_rewards = predicted_reward[: data[INPUT_IDS_CHOSEN_KEY].shape[0]]
            rejected_rewards = predicted_reward[data[INPUT_IDS_CHOSEN_KEY].shape[0] :]
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_chosen_rewards += chosen_rewards.mean().item()
            total_rejected_rewards += rejected_rewards.mean().item()
            total_reward_margin += (chosen_rewards - rejected_rewards).mean().item()
            total_batches += 1

            chosen_texts = tokenizer.batch_decode(data[INPUT_IDS_CHOSEN_KEY])
            rejected_texts = tokenizer.batch_decode(data[INPUT_IDS_REJECTED_KEY])
            # remove padding
            chosen_texts = [item.replace(tokenizer.pad_token, "") for item in chosen_texts]
            rejected_texts = [item.replace(tokenizer.pad_token, "") for item in rejected_texts]
            rewards_rounded = [
                [round(chosen.item(), 4), round(rejected.item(), 4)]
                for chosen, rejected in zip(chosen_rewards, rejected_rewards)
            ]
            correct_prediction = [
                bool((chosen > rejected)) for chosen, rejected in zip(chosen_rewards, rejected_rewards)
            ]
            shared_texts = [
                find_shared_text(chosen_text, rejected_text)
                for chosen_text, rejected_text in zip(chosen_texts, rejected_texts)
            ]
            chosen_response_texts = [
                chosen_text[len(shared_text) :] for chosen_text, shared_text in zip(chosen_texts, shared_texts)
            ]
            rejected_response_texts = [
                rejected_text[len(shared_text) :]
                for rejected_text, shared_text in zip(rejected_texts, shared_texts)
            ]
            table["shared prompt text"].extend(shared_texts)
            table["chosen response text"].extend(chosen_response_texts)
            table["rejected response text"].extend(rejected_response_texts)
            table["chosen reward, rejected reward"].extend(rewards_rounded)
            table["correct prediction"].extend(correct_prediction)

            if "ID" in data:
                table["ID"].extend(data["ID"])

            if max_sampled_texts > 0 and len(table) > max_sampled_texts:
                break

    model.train()
    return {
        "eval/rm/accuracy": total_accuracy / total_batches,
        "eval/rm/loss": total_loss / total_batches,
        "eval/rm/chosen_rewards": total_chosen_rewards / total_batches,
        "eval/rm/rejected_rewards": total_rejected_rewards / total_batches,
        "eval/rm/reward_margin": total_reward_margin / total_batches,
    }, table


def calculate_runtime_args_and_accelerator(args: Args, model_config: ModelConfig) -> Accelerator:
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    # set a unique run name with the current timestamp
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking and accelerator.is_main_process:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    return accelerator


def layer_init(layer: nn.Module, std: float):
    torch.nn.init.normal_(layer.weight, std=std)
    return layer


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    accelerator = calculate_runtime_args_and_accelerator(args, model_config)
    local_seed = args.seed + accelerator.process_index

    # set up experiment tracking and seeds
    all_configs = {}
    all_configs.update(**asdict(args), **asdict(dataset_config), **asdict(model_config))
    if accelerator.is_main_process:
        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=all_configs,
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    # create a tokenizer (pad from right)
    '''
    Extracting the model config and setting the tokenizer
    '''
    config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    )
    if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
        tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    
    # TODO: Remove parquet hardcoding
    data_files = {'train': f'{args.dataset_train_splits}.parquet', 'test': f'{args.dataset_eval_splits}.parquet'}
    dataset = load_dataset('parquet', data_dir=f"{args.dataset_name}", data_files=data_files)
    if args.max_test_dataset is not None:
        dataset['test'] = dataset['test'].shuffle(seed=42).select(range(int( min(args.max_test_dataset, len(dataset['test'])))))  
        
    dataset_processor = PreferencePromptDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    with accelerator.main_process_first():
        dataset = dataset_processor.tokenize(dataset)
        dataset = dataset_processor.filter(dataset)

    # some more runtime logging
    if args.total_episodes is None:
        args.total_episodes = args.num_train_epochs * len(dataset[args.dataset_train_splits])

    args.num_training_steps = args.total_episodes // args.batch_size
    args.eval_freq = max(1, args.total_episodes // args.micro_batch_size // args.num_evals)

    if accelerator.is_main_process:
        pprint([args, dataset_config, model_config])

    # create the model with peft config and optimizer
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        num_labels=1
    )
    #model.config.pad_token_id = tokenizer.pad_token_id (incorrect if you are not resizing embeddings)

    if model_config.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
        )
        model.add_adapter(peft_config)

    if args.resize_token_embeddings:  # optimize for tensor core
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    disable_dropout_in_model(model)  # see p.3. in https://arxiv.org/pdf/1909.08593
    layer_init(
        model.score, std=1 / np.sqrt(model.config.hidden_size + 1)
    )  # see p. 11 in https://arxiv.org/abs/2009.01325
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_training_steps * args.num_train_epochs,
    )
    data_collator = SimplePreferenceCollator(pad_token_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset[args.dataset_train_splits],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        dataset[args.dataset_eval_splits],
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False
    )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    torch.manual_seed(local_seed)

    # set up the metrics and initial states
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    local_metrics = torch.zeros((5,), device=device)
    chosen_rewards = torch.zeros((args.gradient_accumulation_steps,), device=device)
    rejected_rewards = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_margin = torch.zeros((args.gradient_accumulation_steps,), device=device)
    training_step = 0
    gradient_accumulation_idx = 0
    episode = 0
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()

    # training loop
    for epoch_id in range(args.num_train_epochs):
        for data in dataloader:
            episode += args.micro_batch_size
            training_step += 1
            query_responses = torch.cat((data[INPUT_IDS_CHOSEN_KEY], data[INPUT_IDS_REJECTED_KEY]), dim=0)
            with accelerator.accumulate(model):
                _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
                chosen_reward = predicted_reward[: data[INPUT_IDS_CHOSEN_KEY].shape[0]]
                rejected_reward = predicted_reward[data[INPUT_IDS_CHOSEN_KEY].shape[0] :]
                accuracy = (chosen_reward > rejected_reward).float().mean()
                loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                losses[gradient_accumulation_idx] = loss
                accuracies[gradient_accumulation_idx] = accuracy
                chosen_rewards[gradient_accumulation_idx] = chosen_reward.mean()
                rejected_rewards[gradient_accumulation_idx] = rejected_reward.mean()
                reward_margin[gradient_accumulation_idx] = (chosen_reward - rejected_reward).mean()
                gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
                if training_step % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    local_metrics[0] = accuracies.mean()
                    local_metrics[1] = losses.mean()
                    local_metrics[2] = chosen_rewards.mean()
                    local_metrics[3] = rejected_rewards.mean()
                    local_metrics[4] = reward_margin.mean()
                    global_metrics = accelerator.reduce(local_metrics, reduction="mean").tolist()

                    metrics = {
                        "episode": episode,
                        "epoch": episode / len(dataset[args.dataset_train_splits]),
                        "train/rm/accuracy": global_metrics[0],
                        "train/rm/loss": global_metrics[1],
                        "train/rm/chosen_rewards": global_metrics[2],
                        "train/rm/rejected_rewards": global_metrics[3],
                        "train/rm/reward_margin": global_metrics[4],
                        "train/rm/lr": scheduler.get_last_lr()[0],
                    }
                    if accelerator.is_main_process:
                        print_rich_single_line_metrics(metrics)
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, episode)

            # (optionally) evaluate the model
            if args.num_evals > 0 and training_step > 1 and training_step % args.eval_freq == 0:
                eval_metrics, table = evaluate(model, eval_dataloader, tokenizer, max_sampled_texts=10)
                for key in table:
                    table[key] = gather_object(table[key])
                df = pd.DataFrame(table)
                os.makedirs(os.path.dirname(args.eval_dir), exist_ok=True)
                df.to_csv(f'{args.eval_dir}/eval_epoch{epoch_id}.csv')
                if accelerator.is_main_process:
                    print_rich_single_line_metrics(eval_metrics)
                    for key, value in eval_metrics.items():
                        writer.add_scalar(key, value, episode)
                    if args.with_tracking:
                        wandb.log({"preference_sample_texts": wandb.Table(dataframe=df)})
                    else:
                        print_rich_table(df)

            # save model
            if (training_step % args.save_freq == 0):
                save_dir = f"{args.output_dir}/checkpoint-{training_step}"
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                print(f"***** Saving model to {save_dir} *****")
                if model_config.use_peft:
                    model.save_pretrained(args.output_dir)
                    accelerator.wait_for_everyone()  # Ensure synchronization
                else:
                    # original_tokenizer = AutoTokenizer.from_pretrained(
                    #     model_config.model_name_or_path, revision=model_config.model_revision
                    # )
                    save_with_accelerate(
                        accelerator,
                        model,
                        tokenizer,
                        save_dir,
                    )

        if args.push_to_hub:
            push_folder_to_hub(
                accelerator,
                args.output_dir,
                args.hf_repo_id,
                args.hf_repo_revision,
            )
    save_dir = f"{args.output_dir}/checkpoint-{training_step}"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    print(f"***** Saving model to {save_dir} *****")
    if model_config.use_peft:
        model.save_pretrained(args.output_dir)
        accelerator.wait_for_everyone()  # Ensure synchronization
    else:
        # original_tokenizer = AutoTokenizer.from_pretrained(
        #     model_config.model_name_or_path, revision=model_config.model_revision
        # )
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            save_dir,
        )

if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
