from typing import List, Dict, Any, Tuple

class Critic:
    """
    A base class representing a critic that evaluates and assigns scores to 
    reason-action pairs based on a history of observations, reasons, and actions.
    """
    
    def name(self) -> str:
        pass
      
    def score_reason_action(
        self, 
        task: str, 
        observation: str, 
        candidate_actions: List[str], 
        reason: str, 
        action: str
    ) -> float:
        """
        Assigns a score to a given reason-action pair based on the task, observation, 
        and candidate actions.

        Args:
            task (str): The task that the agent is instructed to perform.
            observation (Anstry): The current agent observation.
            candidate_actions (List[str]): A list of possible actions the agent can take.
            reason (str): The explanation or justification for taking the action.
            action (str): The action chosen by the agent.

        Returns:
            float: A numerical score representing the quality of the reason-action pair.
        """
        pass

    def score_reason_action_batch(self, queries: List[Dict]) -> List[float]:
        """
        Assigns scores to a batch of reason-action pairs.

        Args:
            queries (List[Dict]): A list of query dictionaries, each containing:
                - "task": The task the agent is performing.
                - "observation": The agent's current observation.
                - "candidate_actions": Possible actions the agent can take.
                - "observation_action_history": The agent's past interaction history.
                - "reason": The justification for the action.
                - "action": The action taken by the agent.

        Returns:
            List[float]: A list of scores, one for each query.
        """
        pass

    def score_state_batch(self, queries: List[Dict]) -> List[float]:
        """
        Assigns scores to a batch of state representations, evaluating the desirability 
        or utility of each state.

        Args:
            queries (List[Dict]): A list of query dictionaries containing state information.

        Returns:
            List[float]: A list of scores, one per query.
        """
        pass
