import torch
from agent_prm.critics.critic import Critic
from jinja2 import Template
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from typing import Callable, List, Tuple, Dict, Any, Optional
from open_instruct.model_utils import get_reward

# Padding utility
def pad_sequences(sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
    """Pads a list of sequences to the same length."""
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        padded_sequences.append(seq + [pad_token_id] * pad_length)
    return torch.tensor(padded_sequences)

class OpenInstructCritic(Critic):
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 batch_limit = None) -> None:
        self.model_id = model_id
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=1)
        self.verbose = verbose
        self.debug = debug
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        self.batch_limit = batch_limit
        
    def name(self) -> str:
        return self.model_id
    
    def score_reason_action_batch(self, queries: List[Dict]) -> List[float]:
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
    
        ## apply chattemplate and tokenize conversations
        tokenized_conversations = self.tokenizer.apply_chat_template(conversations)

        batch_limit = self.batch_limit if self.batch_limit is not None else len(tokenized_conversations)
        # Batch data according to batch_limit and apply padding
        all_rewards = []
        for i in range(0, len(tokenized_conversations), batch_limit):
            batch = tokenized_conversations[i:i + batch_limit]
            
            # Apply padding to the batch
            padded_batch = pad_sequences(batch, self.tokenizer.pad_token_id)
            
            # Compute rewards for the padded batch
            _, predicted_reward, _ = get_reward(
                self.model,
                padded_batch,  
                self.tokenizer.pad_token_id,
                0
            )
            
            # Collect rewards for the current batch
            all_rewards.extend(predicted_reward.tolist())
        
        return all_rewards
        
