from typing import List, Dict, Any, Tuple
from agent_prm.agents.agent import Agent
import random

def allocate_evenly(N, M):
    if M < N:
        raise ValueError("M must be >= N")
    base, remainder = divmod(M, N)
    return [base + 1 if i < remainder else base for i in range(N)]


class MixtureAgent(Agent):
    def __init__(self,
                 agents,
                 is_low_var = True,
                 verbose: int = 0, 
                 debug: bool = False):
        self.agents = agents
        self.is_low_var = is_low_var
        self.verbose = verbose
        self.debug = debug
            
    def name(self) -> str:
        return "mix-"+" ".join(str(agent.name()) for agent in self.agents)

    def predict_reason_action(self, task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              observation_action_history: List[Dict]) -> Tuple[str, str]:
        agent = random.choice(self.agents)
        return agent.predict_reason_action(task, observation, candidate_actions, observation_action_history)
        

    def predict_reason_action_batch(self, queries: List[Dict], num_responses: int) -> List[Tuple[str, str]]:
        if self.is_low_var:
            assert num_responses >= len(self.agents)
            allocation = allocate_evenly(len(self.agents), num_responses)
            reason_actions_all_queries_combined = None
            
            for agent_idx, agent in enumerate(self.agents):
                reason_actions_all_queries = agent.predict_reason_action_batch(queries=queries, num_responses=allocation[agent_idx])
                
                # Add the name of the agent for logging purposes
                for reason_actions_per_query in reason_actions_all_queries:
                    for reason_action in reason_actions_per_query:
                        reason_action["agent"] = agent.name()
                
                # Initialize the combined list as a list of empty lists
                if reason_actions_all_queries_combined is None:
                    reason_actions_all_queries_combined = [[] for _ in range(len(reason_actions_all_queries))]
                
                for query_idx in range(len(reason_actions_all_queries_combined)):
                    reason_actions_all_queries_combined[query_idx].extend(reason_actions_all_queries[query_idx])
            
            return reason_actions_all_queries_combined
        else:
            agent = random.choice(self.agents)
            return agent.predict_reason_action_batch(queries=queries, num_responses=num_responses)
