import time
import random
from typing import List, Dict
from agent_prm.agents.sglang_server_agent import SGLangServerAgent
from agent_prm.agents.mixture_agent import MixtureAgent
from agent_prm.utils.parser import parse_reason_and_action_alfworld

# Mock query
sample_query = {
    "task": "put a hot cup in cabinet",
    "observation": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a {objects}.\n\nYour task is to: put a hot cup in cabinet.",
    "candidate_actions": ['go to cabinet 1', 'go to cabinet 2', 'go to cabinet 3', 'go to cabinet 4', 'go to cabinet 5', 'go to cabinet 6', 'go to coffeemachine 1', 'go to countertop 1', 'go to countertop 2', 'go to countertop 3', 'go to drawer 1', 'go to drawer 2', 'go to drawer 3', 'go to fridge 1', 'go to garbagecan 1', 'go to microwave 1', 'go to shelf 1', 'go to shelf 2', 'go to shelf 3', 'go to sinkbasin 1', 'go to stoveburner 1', 'go to stoveburner 2', 'go to stoveburner 3', 'go to stoveburner 4', 'go to toaster 1', 'inventory', 'look'],
    "observation_action_history": [
        {"observation": f"observation_{i+1}", "action": f"action_{i+1}"} for i in range(10)
    ],
}

# List of objects for randomization
random_objects = [
    "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1",
    "a coffeemachine 1, a countertop 1, a countertop 2, a countertop 3",
    "a drawer 1, a drawer 2, a drawer 3, a fridge 1",
    "a garbagecan 1, a microwave 1, a shelf 1, a shelf 2, a shelf 3",
    "a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, a toaster 1"
]

def create_batch(query: Dict, batch_size: int) -> List[Dict]:
    """Create a batch of queries with randomized parts of the observation."""
    batch = []
    for _ in range(batch_size):
        randomized_query = query.copy()
        randomized_objects = random.choice(random_objects)
        randomized_query["observation"] = randomized_query["observation"].format(objects=randomized_objects)
        batch.append(randomized_query)
    return batch

# Define the batch size for testing
batch_size = 16 
num_responses = 16


# Instantiate the Agent class (assuming it's already implemented)
agent = SGLangServerAgent(
            model_id="leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter0",
            server_url="http://localhost:30060/",
            prompt_template_file="prompts/alfworld/alfworld_template.j2",
            verbose=False,
            debug=False,
            parse_reason_action_fn=parse_reason_and_action_alfworld,
            temperature=0.3
        )

# policy0 = SGLangServerAgent(
#             model_id="leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter0",
#             server_url="http://localhost:30000/",
#             prompt_template_file="prompts/alfworld/alfworld_template.j2",
#             verbose=False,
#             debug=False,
#             parse_reason_action_fn=parse_reason_and_action_alfworld,
#             temperature=0.3
#         )
# policy1 = SGLangServerAgent(
#             model_id="leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter1",
#             server_url="http://localhost:30060/",
#             prompt_template_file="prompts/alfworld/alfworld_template.j2",
#             verbose=False,
#             debug=False,
#             parse_reason_action_fn=parse_reason_and_action_alfworld,
#             temperature=0.3
#         )

# agent = MixtureAgent(
#     generators = [policy0, policy1],
#     is_low_var=True,
#     verbose=False,
#     debug=False
#     )

# Create the batch
queries = create_batch(sample_query, batch_size)

# Measure execution time
start_time = time.time()
reason_actions_all_queries = agent.predict_reason_action_batch(queries, num_responses=num_responses)
end_time = time.time()

# Results
for query_idx, reason_actions_per_query in enumerate(reason_actions_all_queries):
    print(f"Query #{query_idx}")
    for response_idx, reason_actions in enumerate(reason_actions_per_query):
        print(f"Response #{response_idx}")
        print(f"Reason: {reason_actions['reason']}")
        print(f"Action: {reason_actions['action']}")

# Output results
print(f"Processed batch of size {batch_size} in {end_time - start_time:.2f} seconds.")

# print(f"Results: {results[:5]}")  # Print first few results to verify
