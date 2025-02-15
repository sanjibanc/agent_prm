from jinja2 import Template
from agent_prm.agents.agent import Agent
from typing import Callable, List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer
import requests
from tqdm import tqdm

class SGLangServerAgent(Agent):
    """
    An agent that uses sglang and interacts with an sglang server to predict reasons and actions
    based on task observations and candidate actions. 
    """

    def __init__(self, 
                 model_id: str,
                 server_url: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None,
                 max_tokens=256,
                 temperature=0.3,
                 batch_limit=None) -> None:
        """
        Initializes the SGLangServerAgent with the server URL, prompt template, verbosity, and optional debug settings.
        
        Args:
            server_url: The URL of the sglang server (e.g., "http://localhost:30000/").
            prompt_template_file: Path to a Jinja2 template file for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the generated response to extract reason and action.
        """
        self.model_id = model_id
        self.server_url = server_url.rstrip('/') + '/generate'
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_limit = batch_limit

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def name(self) -> str:
        return self.model_id

    def predict_reason_action(self, 
                              task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              observation_action_history: List[Dict]) -> Tuple[str, str]:
        """
        Predict reason and action given task, observation and candidate_actions. 

        Args:
            task: The task the agent is performing.
            observation: The current observation or input the agent is reacting to.
            candidate_actions: A list of possible actions the agent can take.
            reward: An optional reward signal from prior actions (default is an empty string).
        
        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """

        observation_action_history = [
            {"observation": entry["observation"], "action": entry["action"]}
            for entry in self.observation_action_history
        ]
        input_data = {
            "mode": "input",
            "task": task,
            "observation_action_history": observation_action_history,
            "observation": observation,
            "candidate_actions": candidate_actions,
        }

        input_prompt = self.prompt_template.render(**input_data)
        conversation = [{"role": "user", "content": input_prompt}] 
        input_text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        data = {"model": self.model_id, 
                      "text": input_text,
                      "sampling_params": {
                          "temperature": self.temperature,
                          "max_new_tokens": self.max_tokens,},
                      }
        responses = requests.post(self.server_url, json=data).json()
        generated_text = responses["text"]

        reason, action = self.parse_reason_action_fn(generated_text)

        if self.verbose > 0:
            if self.verbose > 1:
                print(f"\n OBSERVATION: {observation}")
                print(f"\n RESPONSE: {generated_text}")
            print(f"\n OBSERVATION: {observation}")
            print(f"\n CANDIDATE ACTIONS: {candidate_actions}")
            print(f"\n REASON: {reason}")
            print(f"\n ACTION: {action}")

        if self.debug:
            human_input = input()
            if human_input != "c":
                action = human_input
                reason = "None"

        return reason, action
    
    def predict_reason_action_batch(self, queries: List[Dict], num_responses: int) -> List[Tuple[str, str]]:
        """
        Return a list of reason_actions of len(queries), each being len(num_responses)
        """
        input_prompts = []
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
            for _ in range(num_responses):
                input_prompts.append(input_prompt)

        conversations = [[{"role": "user", "content": input_prompt}] for input_prompt in input_prompts] # list of lists

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        generated_texts = []
        
        if self.verbose==1:  
            iterator = tqdm(range(0, len(conversations), batch_limit), desc="Querying sglang agent")
        else:
            iterator = range(0, len(conversations), batch_limit)

        for i in iterator:
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False, add_generation_prompt=True)
            data_batch = {"model": self.model_id, 
                          "text": prompts_batch,
                          "sampling_params": {
                              "temperature": self.temperature,
                              "max_new_tokens": self.max_tokens,
                              },
                          }
            responses_batch = requests.post(self.server_url, 
                                            json=data_batch).json()
            generated_texts_batch = [x["text"] for x in responses_batch]
            generated_texts = generated_texts + generated_texts_batch

        reason_actions_all_queries = []
        counter = 0
        for query in queries:
            reason_actions_per_query = []
            for _ in range(num_responses):
                generated_text = generated_texts[counter]
                counter += 1
                reason, action = self.parse_reason_action_fn(generated_text)
                reason_actions_per_query.append({'reason': reason, 'action': action})

                if self.verbose >= 2:
                    print(f"\n OBSERVATION: {query['observation']}")
                    print(f"\n CANDIDATE ACTIONS: {query['candidate_actions']}")
                    if self.verbose >= 3: 
                        print(f"\n RESPONSE: {generated_text}")
                    print(f"\n REASON: {reason}")
                    print(f"\n ACTION: {action}")
            reason_actions_all_queries.append(reason_actions_per_query)

        if self.debug:
            human_input = input() #can't override a batch of actions

        return reason_actions_all_queries