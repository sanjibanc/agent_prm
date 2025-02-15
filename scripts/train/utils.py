import logging
from typing import Dict, List, Optional, Union
import copy 

import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from open_instruct.dataset_processor import DatasetProcessor, get_num_proc
from jinja2 import Template

logging.basicConfig(level=logging.INFO)


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
LABEL_KEY = "label"

INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"
PROMPT_KEY = 'prompt'
COMPLETION_KEY = 'completion'
TERMINAL_KEY = "terminal"

# FLAGS for Preference dataset
INPUT_IDS_CHOSEN_KEY = 'input_ids_chosen'
INPUT_IDS_REJECTED_KEY = 'input_ids_rejected'

APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130

PROMPT_CURRENT_KEY = 'prompt_current'
PROMPT_FUTURE_KEY = 'prompt_future'
COMPLETION_CURRENT_KEY = 'completion_current'
COMPLETION_FUTURE_KEY = 'completion_future'
INPUT_IDS_PROMPT_CURRENT_KEY = "input_ids_prompt_current"
INPUT_IDS_PROMPT_FUTURE_KEY = "input_ids_prompt_future"

PROMPT_EXPLORATION_KEY = 'prompt_exploration'
INPUT_IDS_PROMPT_EXPLORATION_KEY = "input_ids_prompt_exploration"

class SFTGroundTruthDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
            row[LABELS_KEY] = labels
            row[GROUND_TRUTHS_KEY] = row[self.config.ground_truths_key]
            row[DATASET_SOURCE_KEY] = row[self.config.dataset_source_key]
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class SFTPromptGroundTruthDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            input_data = {'mode': 'input',
                            'observation': row['state']['observation'],
                            'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                            'task': row['state']['task'],
                            'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)

            messages = [{"role": "user", "content": row[PROMPT_KEY]}]
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(messages)
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_KEY])] = [-100] * len(row[INPUT_IDS_KEY])
            row[LABELS_KEY] = labels

            output_data = {'mode': 'output', 
                           'reason': row['reason_action']['reason'], 
                           'action': row['reason_action']['action']}
            row[COMPLETION_KEY] = prompt_template.render(**output_data)
            messages = [{"role": "assistant", "content": prompt_template.render(**output_data)}]
            row[GROUND_TRUTHS_KEY] = self.tokenizer.apply_chat_template(messages)
            
            row[DATASET_SOURCE_KEY] = "alfworld"

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )
    
class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
            row[LABELS_KEY] = labels
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTPromptDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            input_data = {'mode': 'input',
                            'observation': row['state']['observation'],
                            'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                            'task': row['state']['task'],
                            'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)

            messages = [{"role": "user", "content": row[PROMPT_KEY]}]
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(messages)
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            # row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_KEY])] = [-100] * len(row[INPUT_IDS_KEY])
            row[LABELS_KEY] = labels

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )
    
    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class SFTPromptExplorationDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            input_data = {'mode': 'input',
                            'observation': row['state']['observation'],
                            'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                            'task': row['state']['task'],
                            'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)

            messages = [{"role": "user", "content": row[PROMPT_KEY]}]
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(messages)
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            # row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            
            # Exploratoon prompt
            with open("prompts/alfworld/alfworld_exploration_template.j2", "r") as file:
                prompt_exploration_template = Template(file.read())
            row[PROMPT_EXPLORATION_KEY] = prompt_exploration_template.render(**input_data)
            messages_exploration = [{"role": "user", "content": row[PROMPT_EXPLORATION_KEY]}]
            row[INPUT_IDS_PROMPT_EXPLORATION_KEY] = self.tokenizer.apply_chat_template(messages_exploration, add_generation_prompt=True)

            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_KEY])] = [-100] * len(row[INPUT_IDS_KEY])
            row[LABELS_KEY] = labels

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )
    
    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class SimpleGenerateWithExplorationCollator:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        
        input_id_dict = {}
        for prompt_key in [INPUT_IDS_PROMPT_KEY, INPUT_IDS_PROMPT_EXPLORATION_KEY]:            
            max_length = -1
            for i in range(len(batch)):
                max_length = max(max_length, len(batch[i][prompt_key]))
            assert max_length > 0, "the dataset is empty"

            # Initialize lists to store padded sequences and attention masks
            padded_sequences = []

            for i in range(len(batch)):
                # Calculate padding length
                pad_length = max_length - len(batch[i][prompt_key])

                # Pad from the left
                padding = [self.pad_token_id] * pad_length
                padded_sequence = padding + batch[i][prompt_key]
                padded_sequences.append(padded_sequence)

            # Convert to tensors
            padded_sequences = torch.tensor(padded_sequences)
            
            input_id_dict[prompt_key] = padded_sequences

        return input_id_dict
        
class BinaryDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            '''
            FIXME: What should be done with the Attention Mask? 
            '''
            # row[ATTENTION_MASK_KEY] = [1] * len(row['rollouts'])
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                row[PROMPT_KEY] + [row[COMPLETION_KEY]]
            )
            # messages = row['prompt']
            # messages.extend(row['completion'])
            
            # row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages)
            # row[INPUT_PROMPT_KEY] = row['prompt']
            # row[LABEL_KEY] = row['label']
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        '''
        FIXME: Clean this function !
        '''
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class BinaryPromptDatasetProcessor(DatasetProcessor):
    # def __init__(self, prompt_template_file="prompts/alfworld/alfworld_template.j2"):
    #     super(DatasetProcessor, self).__init__()
    #     with open(prompt_template_file, "r") as file:
    #         self.prompt_template = Template(file.read())

    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            
            input_data = {'mode': 'input',
                          'observation': row['state']['observation'],
                          'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                          'task': row['state']['task'],
                          'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)

            output_data = {'mode': 'output', 'reason': row['reason_action']['reason'], 'action': row['reason_action']['action']}
            row[COMPLETION_KEY] = prompt_template.render(**output_data)

            messages = [{"role": "user", "content": row[PROMPT_KEY]},
                        {"role": "assistant", "content": row[COMPLETION_KEY]}]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages)
            row[LABEL_KEY] = row['qestimate']

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        '''
        FIXME: Clean this function !
        '''
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class ValueModelDatasetProcessor(DatasetProcessor):

    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            
            input_data = {'mode': 'input',
                          'observation': row['state']['observation'],
                          'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                          'task': row['state']['task'],
                          'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)

            row[COMPLETION_KEY] = ""

            messages = [{"role": "user", "content": row[PROMPT_KEY]}]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(messages)
            row[LABEL_KEY] = row['vestimate']

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        '''
        FIXME: Clean this function !
        '''
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

class IQLearnDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            
            input_data_current = {'mode': 'input',
                          'observation': row['state']['observation'],
                          'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                          'task': row['state']['task'],
                          'observation_action_history': row['state']['history']}
            row[PROMPT_CURRENT_KEY] = prompt_template.render(**input_data_current)

            output_data_current = {'mode': 'output', 'reason': row['reason_action']['reason'], 'action': row['reason_action']['action']}
            row[COMPLETION_CURRENT_KEY] = prompt_template.render(**output_data_current)

            messages_current = [{"role": "user", "content": row[PROMPT_CURRENT_KEY]},
                        {"role": "assistant", "content": row[COMPLETION_CURRENT_KEY]}]            
            row[INPUT_IDS_PROMPT_CURRENT_KEY] = self.tokenizer.apply_chat_template(messages_current)
            
            row[LABEL_KEY] = row['label']
            row[TERMINAL_KEY] = row['terminal']
            
            if not row['terminal']:
                input_data_future = {'mode': 'input',
                            'observation': row['next_state']['observation'],
                            'candidate_actions': row['next_state']['candidate_actions'] if ('candidate_actions' in row['next_state']) else "",
                            'task': row['next_state']['task'],
                            'observation_action_history': row['next_state']['history']}
                row[PROMPT_FUTURE_KEY] = prompt_template.render(**input_data_future)
                
                output_data_future = {'mode': 'output', 'reason': row['next_reason_action']['reason'], 'action': row['next_reason_action']['action']}
                row[COMPLETION_FUTURE_KEY] = prompt_template.render(**output_data_future)
                            
                messages_future = [{"role": "user", "content": row[PROMPT_FUTURE_KEY]},
                            {"role": "assistant", "content": row[COMPLETION_FUTURE_KEY]}] 
                row[INPUT_IDS_PROMPT_FUTURE_KEY] = self.tokenizer.apply_chat_template(messages_future)
            else:
                # copy over current to make collators etc happy
                row[PROMPT_FUTURE_KEY], row[COMPLETION_FUTURE_KEY], row[INPUT_IDS_PROMPT_FUTURE_KEY] = row[PROMPT_CURRENT_KEY], row[COMPLETION_CURRENT_KEY], row[INPUT_IDS_PROMPT_CURRENT_KEY]
            
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        def filter_fn(row):
            if self.config.max_prompt_token_length is not None:
                if len(row[INPUT_IDS_PROMPT_CURRENT_KEY]) > self.config.max_prompt_token_length:
                    return False
                if (not row[TERMINAL_KEY]) and  len(row[INPUT_IDS_PROMPT_FUTURE_KEY]) > self.config.max_prompt_token_length:
                    return False
            return True


        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_CURRENT_KEY,
                INPUT_IDS_PROMPT_FUTURE_KEY
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )

# class PreferenceDatasetProcessor(DatasetProcessor):
#     def tokenize(self, dataset: Union[Dataset, DatasetDict]):
#         def tokenize_fn(row):

#             row[INPUT_IDS_CHOSEN_KEY] = self.tokenizer.apply_chat_template(
#                 row[PROMPT_KEY] + [row[self.config.preference_chosen_key]]
#             )
#             row[INPUT_IDS_REJECTED_KEY] = self.tokenizer.apply_chat_template(
#                 row[PROMPT_KEY] + [row[self.config.preference_rejected_key]]
#             )
#             row[PROMPT_KEY] = self.tokenizer.apply_chat_template(
#                 row[PROMPT_KEY],
#                 add_generation_prompt=True,
#             )
#             return row

#         return dataset.map(
#             tokenize_fn,
#             num_proc=get_num_proc(
#                 len(dataset),
#                 self.config.num_proc,
#                 APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
#             ),
#             load_from_cache_file=self.config.load_from_cache_file,
#         )

#     def filter(self, dataset: Union[Dataset, DatasetDict]):
#         def filter_fn(row):
#             return (
#                 len(row[PROMPT_KEY]) <= self.config.max_prompt_token_length
#                 if self.config.max_prompt_token_length is not None
#                 else (
#                     True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
#                     if self.config.max_token_length is not None
#                     else (
#                         True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
#                         if self.config.max_token_length is not None
#                         else True
#                     )
#                 )
#             )

#         filtered_dataset = dataset.filter(
#             filter_fn,
#             num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
#             load_from_cache_file=self.config.load_from_cache_file,
#         )
#         if isinstance(dataset, DatasetDict):
#             for key in dataset:
#                 filtered_count = len(dataset[key]) - len(filtered_dataset[key])
#                 total_count = len(dataset[key])
#                 percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
#                 logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
#         return filtered_dataset

#     def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
#         return super().get_token_length_stats(
#             features=[
#                 INPUT_IDS_PROMPT_KEY,
#                 INPUT_IDS_CHOSEN_KEY,
#                 INPUT_IDS_REJECTED_KEY,
#             ],
#             dataset=dataset,
#         )

#     def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
#         return super().get_token_length_visualization(
#             features=[
#                 INPUT_IDS_PROMPT_KEY,
#                 INPUT_IDS_CHOSEN_KEY,
#                 INPUT_IDS_REJECTED_KEY,
#             ],
#             dataset=dataset,
#             save_path=save_path,
#             bins=bins,
#         )

class PreferencePromptDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            with open("prompts/alfworld/alfworld_template.j2", "r") as file:
                prompt_template = Template(file.read())
            
            input_data = {'mode': 'input',
                          'observation': row['state']['observation'],
                          'candidate_actions': row['state']['candidate_actions'] if ('candidate_actions' in row['state']) else "",
                          'task': row['state']['task'],
                          'observation_action_history': row['state']['history']}
            row[PROMPT_KEY] = prompt_template.render(**input_data)
            prompt_messages = [{"role": "user", "content": row[PROMPT_KEY]}]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(prompt_messages)

            chosen_data = {'mode': 'output', 'reason': row['reason_action_chosen']['reason'], 'action': row['reason_action_chosen']['action']}
            chosen_completion = prompt_template.render(**chosen_data)
            chosen_messages = [{"role": "user", "content": row[PROMPT_KEY]},
                               {"role": "assistant", "content": chosen_completion}]
            row[INPUT_IDS_CHOSEN_KEY] = self.tokenizer.apply_chat_template(chosen_messages)
            
            rejected_data = {'mode': 'output', 'reason': row['reason_action_rejected']['reason'], 'action': row['reason_action_rejected']['action']}
            rejected_completion = prompt_template.render(**rejected_data)
            rejected_messages = [{"role": "user", "content": row[PROMPT_KEY]},
                               {"role": "assistant", "content": rejected_completion}]
            row[INPUT_IDS_REJECTED_KEY] = self.tokenizer.apply_chat_template(rejected_messages)

            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        '''
        FIXME: Clean this function !
        '''
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )
        
class SimpleBinaryCollator:
    '''
    Copy of SimplePreferenceCollator in open_instruct and adpated to data columns we are using.
    '''
    def __init__(self, pad_token_id: int):
        """Simple collator for preference dataset (always pad from the RIGHT)"""
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, int]]):
        """the input will have input_ids_chosen, input_ids_rejected"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the right
            padding = [self.pad_token_id] * pad_length
            padded_sequence = batch[i][INPUT_IDS_PROMPT_KEY] + padding
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences_chosen = torch.tensor(padded_sequences)
        rewards = torch.tensor([batch[i][LABEL_KEY] for i in range(len(batch))]).float()

        # Convert to unique prompt keys list
        prompt_keys = [batch[i][PROMPT_KEY] for i in range(len(batch))]
        completion_keys = [batch[i][COMPLETION_KEY] for i in range(len(batch))]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences_chosen,
            LABEL_KEY: rewards,
            PROMPT_KEY: prompt_keys,
            COMPLETION_KEY: completion_keys
        }
        
class IQLearnCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, int]]):
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_CURRENT_KEY]))
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_FUTURE_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences_current = []
        for i in range(len(batch)):
            # Pad current
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_CURRENT_KEY])
            padding = [self.pad_token_id] * pad_length
            padded_sequence_current = batch[i][INPUT_IDS_PROMPT_CURRENT_KEY] + padding
            padded_sequences_current.append(padded_sequence_current)

        padded_sequences_future = []
        for i in range(len(batch)):
            # Pad future
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_FUTURE_KEY])
            padding = [self.pad_token_id] * pad_length
            padded_sequence_future = batch[i][INPUT_IDS_PROMPT_FUTURE_KEY] + padding
            padded_sequences_future.append(padded_sequence_future)

        # Convert to tensors
        padded_sequences_current_tensor = torch.tensor(padded_sequences_current)
        padded_sequences_future_tensor = torch.tensor(padded_sequences_future)
        rewards = torch.tensor([batch[i][LABEL_KEY] for i in range(len(batch))]).float()
        terminals = torch.tensor([batch[i][TERMINAL_KEY] for i in range(len(batch))]).bool()

        # Convert to unique prompt keys list
        prompt_current_keys = [batch[i][PROMPT_CURRENT_KEY] for i in range(len(batch))]
        prompt_future_keys = [batch[i][PROMPT_FUTURE_KEY] for i in range(len(batch))]
        completion_current_keys = [batch[i][COMPLETION_CURRENT_KEY] for i in range(len(batch))]
        completion_future_keys = [batch[i][COMPLETION_FUTURE_KEY] for i in range(len(batch))]

        return {
            INPUT_IDS_PROMPT_CURRENT_KEY: padded_sequences_current_tensor,
            INPUT_IDS_PROMPT_FUTURE_KEY: padded_sequences_future_tensor,
            LABEL_KEY: rewards,
            TERMINAL_KEY: terminals,
            PROMPT_CURRENT_KEY: prompt_current_keys,
            COMPLETION_CURRENT_KEY: completion_current_keys,
            PROMPT_FUTURE_KEY: prompt_future_keys,
            COMPLETION_FUTURE_KEY: completion_future_keys,
        }

class SimpleGenerateCollatorWithGroundTruth:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the left
            padding = [self.pad_token_id] * pad_length
            padded_sequence = padding + batch[i][INPUT_IDS_PROMPT_KEY]
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences)

        # ground truths
        ground_truths = [x[GROUND_TRUTHS_KEY] for x in batch]

        # datasets
        datasets = [x[DATASET_SOURCE_KEY] for x in batch]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences,
            GROUND_TRUTHS_KEY: ground_truths,
            DATASET_SOURCE_KEY: datasets,
        }