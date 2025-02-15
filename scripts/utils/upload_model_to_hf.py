import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator
from huggingface_hub import HfApi

def upload_normal_model(input_model, output_model):
    model = AutoModelForCausalLM.from_pretrained(input_model)
    tokenizer = AutoTokenizer.from_pretrained(input_model)
    
    repo_id = f"{output_model}"
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    
    print(f"Successfully uploaded {input_model} to Hugging Face Hub under organization {output_model}")

def load_and_merge_peft_model(input_model, output_model, upload_adapters_only=False):
    config_file = os.path.join(input_model, "adapter_config.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found in the directory: {input_model}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    base_model_name = config.get('base_model_name_or_path')
    
    if not base_model_name:
        raise ValueError("Base model name not found in the adapter config.")
    
    if not upload_adapters_only:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, input_model)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(input_model, truncation=True, padding=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            input_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(input_model, truncation=True, padding=True)
    
    tmp_save_dir = os.path.join("save/tmp/", output_model)    
    model.save_pretrained(tmp_save_dir, push_to_hub=True, repo_id=f"{output_model}")
    tokenizer.push_to_hub(repo_id=f"{output_model}")
    
    print(f"Successfully uploaded {input_model} to Hugging Face Hub under organization {output_model}")

def push_folder_to_hub_with_accelerator(accelerator, output_dir, hf_repo_id, hf_repo_revision=None, private=True):
    if accelerator.is_main_process:
        hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}" if hf_repo_revision else f"https://huggingface.co/{hf_repo_id}/"
        print(hf_repo_url)

        api = HfApi()
        if not api.repo_exists(hf_repo_id):
            api.create_repo(hf_repo_id, exist_ok=True, private=private)
        if hf_repo_revision:
            api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
        api.upload_folder(
            repo_id=hf_repo_id,
            revision=hf_repo_revision,
            folder_path=output_dir,
            commit_message="upload checkpoint",
            run_as_future=False,
        )
        print(f"🔥 Pushed to {hf_repo_url}")

# Argument parser
parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
parser.add_argument('--input_model', help='Input model path')
parser.add_argument('--output_model', help='Output model path on Hugging Face')
parser.add_argument('--peft', action='store_true', help='Set this flag to upload a PEFT model')
parser.add_argument('--upload_adapters_only', action='store_true', help='Only upload the PEFT adapters without merging the model')
parser.add_argument('--accelerate', action='store_true', help='Use Accelerate for distributed upload')
parser.add_argument('--hf_repo_revision', help='Optional: Hugging Face repo revision (branch)', default=None)
parser.add_argument('--private', action='store_true', help='Set the Hugging Face repository as private', default=True)
args = parser.parse_args()

# Ask user for confirmation
confirm = input(f"Do you want to upload {args.input_model} to {args.output_model}? [y/n] ").strip().lower()
if confirm != 'y':
    print("Upload cancelled by user.")
    exit()

# Upload model based on flags
if args.accelerate:
    accelerator = Accelerator()
    push_folder_to_hub_with_accelerator(
        accelerator=accelerator,
        output_dir=args.input_model,
        hf_repo_id=args.output_model,
        hf_repo_revision=args.hf_repo_revision,
        private=args.private
    )
elif args.peft:
    load_and_merge_peft_model(
        input_model=args.input_model,
        output_model=args.output_model,
        upload_adapters_only=args.upload_adapters_only
    )
else:
    upload_normal_model(
        input_model=args.input_model,
        output_model=args.output_model
    )
