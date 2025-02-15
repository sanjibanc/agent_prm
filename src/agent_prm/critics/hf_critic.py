import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
from jinja2 import Template

# Padding utility
def pad_sequences(sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
    """Pads a list of sequences to the same length."""
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        padded_sequences.append(seq + [pad_token_id] * pad_length)
    return torch.tensor(padded_sequences)

class HFCritic:
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 batch_limit: int = None):
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.batch_limit = batch_limit

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        
        pad_token = "<|eom_id|>"
        self.tokenizer.pad_token = pad_token
        self.model.config.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)

        # Load the prompt template
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

    def name(self) -> str:
        return self.model_id

    def score_reason_action_batch(self, queries: List[Dict]) -> List[float]:
        # Generate prompts from queries
        conversations = []
        for query in queries:
            observation_action_history = [
                {"observation": entry["observation"], "action": entry["action"]}
                for entry in query["observation_action_history"]
            ]
            input_data = {
                "mode": "input",
                "task": query["task"],
                "observation": query["observation"],
                "candidate_actions": query["candidate_actions"],
                "observation_action_history": observation_action_history,      
            }
            input_prompt = self.prompt_template.render(**input_data)

            output_data = {
                "mode": "output",
                "reason": query["reason"],
                "action": query["action"]     
            }
            output_prompt = self.prompt_template.render(**output_data)
            conversations.append([{"role": "user", "content": input_prompt}, {"role": "assistant", "content": output_prompt}])

        
        tokenized_conversations = self.tokenizer.apply_chat_template(conversations)
        import pdb; pdb.set_trace()
        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        # Tokenize conversations
        all_rewards = []
        for i in range(0, len(tokenized_conversations), batch_limit):
            batch = tokenized_conversations[i:i + batch_limit]
            input_ids = pad_sequences(batch, self.tokenizer.pad_token_id)
            attention_mask = input_ids != self.model.config.pad_token_id
            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
            
            logits = outputs[0]
            rewards = logits.detach().cpu().squeeze().tolist()

            # Collect batch rewards
            all_rewards.extend(rewards)

        return all_rewards
