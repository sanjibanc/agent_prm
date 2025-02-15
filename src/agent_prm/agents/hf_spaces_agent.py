from jinja2 import Template
from agent_prm.agents.agent import Agent
from gradio_client import Client
from typing import Callable, List, Tuple, Any, Optional, Dict


class HFSpaceAgent(Agent):
    """
    An agent that interacts with a Hugging Face Space via the Gradio API to predict reasons and actions 
    based on task observations and a history of interactions.
    """
    def __init__(self, 
                 model_id: str,
                 space_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None) -> None:
        """
        Initializes the HFSpaceAgent with a Hugging Face Space ID, a prompt template, verbosity, and optional debug settings.
        
        Args:
            space_id: The identifier for the Hugging Face Space being used.
            prompt_template_file: Path to a Jinja2 template file for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the model's response to extract reason and action.
        """
        self.model_id = model_id
        self.space_id = space_id
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())
        
        self.client = Client(space_id)

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
            observation_action_history (List[Dict]): A history of previous observations and actions.
        
        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """
        observation_action_history = [{'observation': entry['observation'], 'action': entry['action']} for entry in self.observation_action_history]

        input_data = {
            'mode': 'input',
            'task': task,
            'observation_action_history': observation_action_history,
            'observation': observation,
            'candidate_actions': candidate_actions
        }
        input_prompt = self.prompt_template.render(**input_data)

        messages = [
            {"role": "user", "content": input_prompt}
        ]

        response = self.client.predict(
            messages,  
            api_name="/predict"  
        )

        reason, action = self.parse_reason_action_fn(response)
        if self.verbose > 0:
            if self.verbose > 1:
                print(f"\n OBSERVATION: {observation}")
                print(f"\n RESPONSE: {response}")
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
