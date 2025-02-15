from agent_prm.critics.critic import Critic
from jinja2 import Template
from transformers import AutoTokenizer, AutoConfig
from typing import Callable, List, Tuple, Dict, Any, Optional
import requests
from tqdm import tqdm

class SGLangServerCritic(Critic):
    def __init__(self, 
                 model_id: str, 
                 server_url: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None,
                 batch_limit = None) -> None:
        self.server_url = server_url.rstrip('/') + '/classify'
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        self.tokenizer = tokenizer
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

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        scores = []
        for i in range(0, len(conversations), batch_limit):
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False)
            data_batch = {"model": self.model_id, "text": prompts_batch}
            responses_batch = requests.post(self.server_url, json=data_batch).json()
            scores_batch = [x["embedding"][0] for x in responses_batch]
            scores = scores + scores_batch
            
        return scores
    
    def score_state_batch(self, queries: List[Dict]) -> List[float]:
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

            conversations.append([{"role": "user", "content": input_prompt}])      

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        scores = []
        
        if self.verbose==1:  
            iterator = tqdm(range(0, len(conversations), batch_limit), desc="Querying sglang critic")
        else:
            iterator = range(0, len(conversations), batch_limit)

        for i in iterator:
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False)
            data_batch = {"model": self.model_id, "text": prompts_batch}
            responses_batch = requests.post(self.server_url, json=data_batch).json()
            scores_batch = [x["embedding"][0] for x in responses_batch]
            scores = scores + scores_batch
            
        return scores