from jinja2 import Template
from agent_prm.agents.agent import Agent
from typing import Callable, List, Tuple, Any, Optional, Dict

class BestofNAgent(Agent):
    """
    An agent that generates multiple reason-action pairs for a given query and selects the best
    responses using a critic model.
    """
    def __init__(
        self, 
        generator: Agent,
        critic: Any, 
        num_generations: int = 0, 
        verbose: int = 0, 
        debug: bool = False
    ) -> None:
        """
        Initializes the BestofNAgent with a generator, critic, and generation parameters.

        Args:
            generator (Agent): The agent responsible for generating multiple reason-action candidates.
            critic (Any): A scoring mechanism to evaluate generated reason-action pairs.
            num_generations (int, optional): The number of candidate responses to generate per query. Defaults to 0.
            verbose (int, optional): Verbosity level for debug logging. Defaults to 0.
            debug (bool, optional): If True, prompts for human input during debugging. Defaults to False.
        """
        self.generator = generator
        self.critic = critic
        self.num_generations = num_generations
        self.verbose = verbose
        self.debug = debug

    def name(self) -> str:
        return f"critic-{self.critic.name()}-generator-{self.generator.name()}-{self.num_generations}"
    
    def predict_reason_action_batch(self, queries: List[Dict], num_responses: int) -> List[List[Dict]]:
        assert num_responses <= self.num_generations # Cannot return more than generations

        reason_actions_all_queries = self.generator.predict_reason_action_batch(queries=queries, num_responses=self.num_generations)

        # Flatten the queries and their corresponding reason-actions
        flattened_queries_with_reason_action = []
        for query, reason_actions_per_query in zip(queries, reason_actions_all_queries):
            for reason_action in reason_actions_per_query:
                flattened_queries_with_reason_action.append({**query, **reason_action})

        # Compute scores for each (query, reason-action) pair
        scores = self.critic.score_reason_action_batch(queries=flattened_queries_with_reason_action)

        # Attach scores to each reason-action
        counter = 0
        for reason_actions_per_query in reason_actions_all_queries:
            for reason_action in reason_actions_per_query:
                reason_action['score'] = scores[counter]
                counter += 1

        # Sort reason-actions by score and select top responses for each query
        reason_actions_all_queries_top = []
        for reason_actions_per_query in reason_actions_all_queries:
            sorted_reason_actions = sorted(reason_actions_per_query, key=lambda x: x['score'], reverse=True)
            reason_actions_all_queries_top.append(sorted_reason_actions[:num_responses])

        self.agent_log = reason_actions_all_queries
        ## DEBUG PRINTING
        if self.verbose > 0:
            reason_actions_all_queries_sorted = [sorted(batch, key=lambda x: x['score'], reverse=True) for batch in reason_actions_all_queries]
            for query, reason_actions_per_query in zip(queries, reason_actions_all_queries_sorted):
                print(f"\n Observation: {query['observation']}")
                print(f"\n Candidate Actions: {query['candidate_actions']}")
                for idx, reason_action in enumerate(reason_actions_per_query):
                    if idx < num_responses:
                        print(f"\n Score: {reason_action['score']} Reason: {reason_action['reason']} Action: {reason_action['action']}")
                    else:
                        print(f"\n [REJ] Score: {reason_action['score']} Reason: {reason_action['reason']} Action: {reason_action['action']}")
        
        if self.debug:
            human_input = input() 

        return reason_actions_all_queries_top

    def get_log(self):
        return self.agent_log