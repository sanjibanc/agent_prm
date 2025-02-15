from typing import List, Dict, Any, Tuple
from agent_prm.agents.agent import Agent
import random

class SwitchAgent(Agent):
    def __init__(self,
                 agent_rollout,
                 agent_switch,
                 switch_time_horizon,
                 verbose: int = 0, 
                 debug: bool = False):
        self.agent_rollout = agent_rollout
        self.agent_switch = agent_switch
        self.switch_time_horizon = switch_time_horizon
        self.verbose = verbose
        self.debug = debug
        
        self.switch_time = random.randint(0, self.switch_time_horizon)
        print(f"Switch time: {self.switch_time}")
            
    def name(self) -> str:
        return "switch-"+self.agent_rollout.name()+"-"+self.agent_switch.name()

    def reset(self) -> None:
        self.observation_action_history = []
        self.switch_time = random.randint(0, self.switch_time_horizon)
        print(f"Switch time: {self.switch_time}")
        
    def predict_reason_action_batch(self, queries: List[Dict], num_responses: int) -> List[Tuple[str, str]]:
        # Assume for simplicity all queries are at the same timestep
        timestep = None
        for query in queries:
            if timestep == None:
                timestep = len(query["observation_action_history"])
            else:
                assert timestep == len(query["observation_action_history"])
        
        if timestep == self.switch_time:
            return self.agent_switch.predict_reason_action_batch(queries=queries, num_responses=num_responses)
        else:
            return self.agent_rollout.predict_reason_action_batch(queries=queries, num_responses=num_responses)
