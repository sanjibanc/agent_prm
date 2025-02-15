import os
import json
from hashlib import sha256
from tqdm import tqdm


# Function to hash a state based on observation, candidate actions, and history
def hash_state(trajectory, index):
    """
    Create a hash for a state.
    
    :param trajectory: List of dicts containing trajectory information.
    :param index: Index of the current state in the trajectory.
    :return: Hash value as a string.
    """
    history = []
    for i in range(index):
        step = trajectory[i]
        history.append({
            'observation': step['observation'],
            'action': step['action']
        })
    
    current_state = {
        'observation': trajectory[index]['observation'],
        'candidate_actions': trajectory[index]['candidate_actions'],
        'action': trajectory[index]['action'],
        'history': history
    }
    
    return sha256(json.dumps(current_state, sort_keys=True).encode()).hexdigest(), current_state

# Path to folder with JSON files
folder_path = "data/rollout/alfworld/20241221-220918/leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter1"

# Counter for total non-unique states
total_states = 0

# Dictionary to store unique state hashes and their raw info
unique_states = {}

# Iterate through all files in the folder
for file_name in tqdm(os.listdir(folder_path), desc="Processing files"):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        
        # Load JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Check if trajectory field exists
            if 'trajectory' in data:
                trajectory = data['trajectory']
                total_states += len(trajectory)  # Add all states to total count
                
                # Add each state hash to the unique set
                for index in range(len(trajectory)):
                    state_hash, raw_info = hash_state(trajectory, index)
                    unique_states[state_hash] = raw_info

print(f"Total non-unique states across all trajectories: {total_states}")
print(f"Total unique states across all trajectories: {len(unique_states)}")
