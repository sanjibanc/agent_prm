from jinja2 import Template
from agent_prm.agents.agent import Agent
from typing import Callable, List, Tuple, Any, Optional, Dict

class ChatAgent(Agent):
    """
    A specialized agent that utilizes a chat-based generation function (`generate_fn`)
    to predict a reason and an action based on a given task, observation, and candidate actions.
    """
    def __init__(self, model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 generate_fn: Callable[[List[dict], str], Tuple[str, Any]] = None, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None) -> None:
        """
        Initializes the ChatAgent with a model identifier, a prompt template, and optional verbosity/debug settings.
        
        Args:
            model_id: The identifier for the language model being used.
            prompt_template_file: The file path to a Jinja2 template for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            generate_fn: A callable function for generating model responses based on messages.
            parse_reason_action_fn: A callable function for parsing the model response to extract reason and action.
        """
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.generate_fn = generate_fn
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

    def name(self) -> str:
        return self.model_id
    
    def predict_reason_action(
        self, 
        task: str, 
        observation: Any, 
        candidate_actions: List[str], 
        observation_action_history: List[Dict]
    ) -> Tuple[str, str]:
        """
        Predicts a reason and an action based on the given task, observation, and candidate actions.

        Args:
            task (str): The task the agent is performing.
            observation (Any): The current observation that the agent is responding to.
            candidate_actions (List[str]): A list of possible actions the agent can take.
            observation_action_history (List[Dict]): A history of previous observations and actions.

        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """
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
        response, _ = self.generate_fn(messages=messages, model=self.model_id)

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
