import time
import random
from typing import List, Dict
from agent_prm.agents.hf_agent import HFAgent
from agent_prm.utils.parser import parse_reason_and_action_alfworld

# Mock query
sample_query = {
    "task": "put a hot cup in cabinet",
    "observation": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a a coffeemachine 1, a countertop 1, a countertop 2, a countertop 3.\n\nYour task is to: put a hot cup in cabinet.",
    "candidate_actions": ['go to cabinet 1', 'go to cabinet 2', 'go to cabinet 3', 'go to cabinet 4', 'go to cabinet 5', 'go to cabinet 6', 'go to coffeemachine 1', 'go to countertop 1', 'go to countertop 2', 'go to countertop 3', 'go to drawer 1', 'go to drawer 2', 'go to drawer 3', 'go to fridge 1', 'go to garbagecan 1', 'go to microwave 1', 'go to shelf 1', 'go to shelf 2', 'go to shelf 3', 'go to sinkbasin 1', 'go to stoveburner 1', 'go to stoveburner 2', 'go to stoveburner 3', 'go to stoveburner 4', 'go to toaster 1', 'inventory', 'look'],
    "observation_action_history": [{"observation": "observation_1", "action": "action_1"}],
}

agent = HFAgent(model_id= "leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter1",
        prompt_template_file= "prompts/alfworld/alfworld_template.j2",
        verbose= False,
        debug= False,
        parse_reason_action_fn= parse_reason_and_action_alfworld, 
        max_length=6000)

# Measure execution time
start_time = time.time()
reason, action = agent.predict_reason_action(task = sample_query["task"], 
                      observation = sample_query["observation"], 
                      candidate_actions = sample_query["candidate_actions"], 
                      observation_action_history=[])
end_time = time.time()

# Output results
print(f"Processed in {end_time - start_time:.2f} seconds.")
print(f"Reason: {reason} Action: {action}")  # Print first few results to verify
