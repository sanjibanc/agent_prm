import transformers
from trl import DPOConfig
from dataclasses import dataclass, field
from typing import List, Optional
from peft import LoraConfig, PeftConfig
from transformers import BitsAndBytesConfig

@dataclass
class ModelArguments:
    """
    Holds the arguments related to the model configuration, including quantization and PEFT (LoRA) settings.
    """
    model_id: Optional[str] = field(default="meta-llama/Llama-3-8b-Instruct")

    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )

    # peft config
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"
        },
    )

    # quantization config
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "use 8 bit precision for the base model - works only with LoRA"
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "use 4 bit precision for the base model - works only with LoRA"
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(
        default=True, metadata={"help": "use nested quantization"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Extends the Hugging Face `TrainingArguments` class with additional fields specific to fine-tuning models.
    """
    packing: bool = field(default=True, metadata={"help": "pack consecutive sequences"})

    # sft trainer
    max_seq_length: int = field(
        default=1024, metadata={"help": "max seq length of each sample (sft trainer)"}
    )

@dataclass
class DPOTrainingArguments(DPOConfig):
    pass

def get_quantization_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    """
    Generates the quantization configuration for the model based on the provided arguments.

    Args:
        model_args: The model configuration arguments, containing quantization settings.

    Returns:
        An optional `BitsAndBytesConfig` object based on whether quantization is enabled.
    """
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.torch_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_peft_config(model_args: ModelArguments) -> Optional[PeftConfig]:
    """
    Generates the PEFT (LoRA) configuration based on the provided model arguments.

    Args:
        model_args: The model configuration arguments, including LoRA-specific settings.

    Returns:
        An optional `PeftConfig` object if PEFT is enabled, or None otherwise.
    """
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=model_args.lora_task_type,
        base_model_name_or_path=model_args.model_id,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config
