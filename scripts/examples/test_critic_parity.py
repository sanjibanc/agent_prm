import time
import random
from typing import List, Dict
from agent_prm.critics.sglang_server_critic import SGLangServerCritic
from agent_prm.critics.open_instruct_critic import OpenInstructCritic
from agent_prm.critics.hf_critic import HFCritic

# Mock query
sample_query = {
    "task": "put a hot cup in cabinet",
    "observation": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a {objects}.\n\nYour task is to: put a hot cup in cabinet.",
    "candidate_actions": ['go to cabinet 1', 'go to cabinet 2', 'go to cabinet 3', 'go to cabinet 4', 'go to cabinet 5', 'go to cabinet 6', 'go to coffeemachine 1', 'go to countertop 1', 'go to countertop 2', 'go to countertop 3', 'go to drawer 1', 'go to drawer 2', 'go to drawer 3', 'go to fridge 1', 'go to garbagecan 1', 'go to microwave 1', 'go to shelf 1', 'go to shelf 2', 'go to shelf 3', 'go to sinkbasin 1', 'go to stoveburner 1', 'go to stoveburner 2', 'go to stoveburner 3', 'go to stoveburner 4', 'go to toaster 1', 'inventory', 'look'],
    "observation_action_history": [
        {"observation": f"observation_{i+1}", "action": f"action_{i+1}"} for i in range(10)
    ],
    "reason": "The cabinet is the best place to store a cup.",
    "action": "go to cabinet 6"
}

# List of objects for randomization
random_objects = [
    "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1",
    "a coffeemachine 1, a countertop 1, a countertop 2, a countertop 3",
    "a drawer 1, a drawer 2, a drawer 3, a fridge 1",
    "a garbagecan 1, a microwave 1, a shelf 1, a shelf 2, a shelf 3",
    "a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, a toaster 1"
]

def create_batch(query: Dict, batch_size: int, seed: int = None) -> List[Dict]:
    """Create a batch of queries with randomized parts of the observation.

    Args:
        query (Dict): The input query template.
        batch_size (int): The number of queries in the batch.
        seed (int, optional): The seed for reproducibility. Defaults to None.

    Returns:
        List[Dict]: A batch of queries with randomized observations.
    """
    if seed is not None:
        random.seed(seed)
    
    batch = []
    for i in range(batch_size):
        randomized_query = query.copy()
        #randomized_objects = random.choice(random_objects)
        randomized_objects = random_objects[i]
        randomized_query["observation"] = randomized_query["observation"].format(objects=randomized_objects)
        batch.append(randomized_query)
    return batch

# Define the batch size for testing
batch_size = 2 

critics = [
    SGLangServerCritic(
        model_id="rl-llm-agent/Llama-3.2-1B-Instruct-reward-alfworld-iter0-iter1",
        server_url="http://localhost:30030/",
        prompt_template_file="prompts/alfworld/alfworld_reward_template.j2"
    ),
    OpenInstructCritic(
        model_id="rl-llm-agent/Llama-3.2-1B-Instruct-reward-alfworld-iter0-iter1",
        prompt_template_file="prompts/alfworld/alfworld_reward_template.j2",
    ),
    HFCritic(
        model_id="rl-llm-agent/Llama-3.2-1B-Instruct-reward-alfworld-iter0-iter1",
        prompt_template_file="prompts/alfworld/alfworld_reward_template.j2",
    )
]

queries = create_batch(sample_query, batch_size)

for critic in critics:    
    # Measure execution time
    start_time = time.time()
    scores = critic.score_reason_action_batch(queries)
    end_time = time.time()

    # Results
    for query_idx, score in enumerate(scores):
        print(f"Query #{query_idx}")
        print(f"Score: {score}")
