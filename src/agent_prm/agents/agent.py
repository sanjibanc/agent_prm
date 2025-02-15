from typing import List, Dict, Any, Tuple

class Agent:
    """
    A base class for an agent that predicts a reason and an action based on a history of 
    observations, reasons, and actions.
    """
    
    def name(self) -> str:
        """
        Returns the agent's name
        """
        pass

    def predict_reason_action(self, task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              observation_action_history: List[Dict]) -> Tuple[str, str]:
        """
        Predicts a reason and an action given the current task, observation, and action candidates.

        Args:
            task (str): The task the agent is instructed to perform.
            observation (Any): The current agent observation.
            candidate_actions (List[str]): A list of possible actions the agent can take.
            observation_action_history (Any): The agent's past history of observations and actions.

        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """
        pass

    def predict_reason_action_batch(self, queries: List[Dict], num_responses: int) -> List[Tuple[str, str]]:
        """
        Predicts reasons and actions for a batch of queries.

        Args:
            queries (List[Dict]): A list of query dictionaries, each containing:
                - "task": The task the agent is performing.
                - "observation": The agent's current observation.
                - "candidate_actions": Possible actions the agent can take.
                - "observation_action_history": The agent's past interaction history.
            num_responses (int): The number of responses to generate per query.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains:
                - The predicted reason for the action.
                - The predicted action.
        """
        pass

    def get_log(self):
        return None