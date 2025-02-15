from transformers import AutoTokenizer
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

import requests

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)

RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}],
    [{"role": "user", "content": PROMPT}],
]

tokenizer = AutoTokenizer.from_pretrained("rl-llm-agent/Llama-3.2-1B-Instruct-Sft-Alfworld-v0")
prompts = tokenizer.apply_chat_template(CONVS, tokenize=False)

url = "http://localhost:30000/generate"
data = {"model": "rl-llm-agent/Llama-3.2-1B-Instruct-Sft-Alfworld-v0", "text": prompts}

responses = requests.post(url, json=data).json()
import pdb; pdb.set_trace()
for response in responses:
    print_highlight(f"response: {response}")