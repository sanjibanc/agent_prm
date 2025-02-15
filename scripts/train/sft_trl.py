from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig

import multiprocessing
import wandb

from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

@dataclass
class Arguments:
    # data configs
    data_dir: str = field(default=None, metadata={"help": "path to training data"})
    prior_data_dir: str = field(default=None, metadata={"help": "path to prior data"})
    data_dirs: str = field(default=None, metadata={"help": "path to training data directories"})
    wandb_project_name: str = field(default="LLM_RM", metadata={"help": "wandb project name"})

if __name__ == "__main__":
    parser = TrlParser((Arguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project=args.wandb_project_name)

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    peft_config = get_peft_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    # if config.architectures == "LlamaForCausalLM" and model_config.bos_token_id == 128000:
    #     tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    # else:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    if args.data_dir is not None:
        ###### load dataset ######
        train_dataset = load_dataset(
            "json", data_files=f"{args.data_dir}/train.json", split="train"
        ).shuffle(seed=42)

        eval_dataset = load_dataset(
            "json", data_files=f"{args.data_dir}/test.json", split="train"
        ).shuffle(seed=42)

        if args.prior_data_dir is not None:
            prior_train_dataset = load_dataset(
                "json", data_files=f"{args.prior_data_dir}/train.json", split="train"
            ).shuffle(seed=42)

            if len(prior_train_dataset) > len(train_dataset):
                prior_train_dataset = prior_train_dataset.select(range(len(train_dataset)))

            train_dataset = concatenate_datasets([train_dataset, prior_train_dataset])
            train_dataset = train_dataset.shuffle(seed=42)
    elif args.data_dirs is not None:
        ###### Load and merge datasets from multiple directories ######
        data_dirs = args.data_dirs.split(',')

        train_datasets, eval_datasets = [], []
        for data_dir in data_dirs:
            train_datasets.append(load_dataset("json", data_files=f"{data_dir}/train.json", split="train"))
            eval_datasets.append(load_dataset("json", data_files=f"{data_dir}/test.json", split="train"))
        
        N_train = min(len(dataset) for dataset in train_datasets)
        N_eval = min(len(dataset) for dataset in eval_datasets)

        # Initialize merged datasets by selecting equal amounts from each dataset
        train_dataset = concatenate_datasets([
            dataset.shuffle(seed=42).select(range(N_train)) for dataset in train_datasets
        ]).shuffle(seed=42)

        eval_dataset = concatenate_datasets([
            dataset.shuffle(seed=42).select(range(N_eval)) for dataset in eval_datasets
        ]).shuffle(seed=42)
    else:
        raise ValueError("One of the two must be valid: data_dir or data_dirs.")
    
    completion_token = "<|response|>"

    def process(row):
        row["response"][0][
            "content"
        ] = f"{completion_token} {row['response'][0]['content']}"
        row["text"] = tokenizer.apply_chat_template(
            row["prompt"] + row["response"], tokenize=False
        )
        return row

    train_dataset = train_dataset.map(
        process,
        num_proc=32,
        load_from_cache_file=False,
    )
    eval_dataset = eval_dataset.map(
        process,
        num_proc=32,
        load_from_cache_file=False,
    )

    ################
    # Training
    ################
    collator = DataCollatorForCompletionOnlyLM(
        completion_token,
        tokenizer=tokenizer,
        return_tensors="pt")

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator
    )

    trainer.train()

    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=dataset_name)
