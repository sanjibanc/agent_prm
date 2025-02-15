import os
import json
import argparse
import yaml
import random
from hashlib import sha256
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def print_vestimate_histogram(V_target, bins=10):
    vestimates = [entry['vestimate'] for entry in V_target.values()]
    counts, bin_edges = np.histogram(vestimates, bins=bins)
    
    for i in range(len(counts)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {counts[i]}")

def extract_state_reason_action(trajectory, task, t):
    history = []
    for i in range(t):
        step = trajectory[i]
        history.append({
            'observation': step['observation'],
            'action': step['action']
        })
    
    state = {
        'observation': trajectory[t]['observation'],
        'candidate_actions': trajectory[t]['candidate_actions'],
        'history': history,
        'task': task
    }
    reason_action = {
        'reason': trajectory[t]['reason'],
        'action': trajectory[t]['action'],
    }

    return state, reason_action

def update_V(state, state_hash, V_target, outcome_reward, gamma, t):
    if state_hash not in V_target:
        V_target[state_hash] = {
            'state': state,
            'vestimate': 0,
            'count': 0
        }
    current_entry = V_target[state_hash]
    current_vestimate = current_entry['vestimate']
    current_count = current_entry['count']
    
    updated_vestimate = (current_vestimate * current_count + (gamma ** t) * outcome_reward) / (current_count + 1)
    V_target[state_hash]['vestimate'] = updated_vestimate
    V_target[state_hash]['count'] = current_count + 1

def compute_vestimate(rolloutdirs, outputdir, gamma, train_split):
    V_target = {}
    for rolloutdir in rolloutdirs:
        for file_name in tqdm(os.listdir(rolloutdir), desc=f"Processing files in {rolloutdir}"):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(rolloutdir, file_name)
                
            # Load JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            if 'trajectory' not in data:
                continue
            
            trajectory = data['trajectory']
            outcome_reward = 2 * trajectory[-1]['score'] - 1  # transform from [-1, 1]
            for t in range(len(trajectory) - 1, -1, -1):
                state, _ = extract_state_reason_action(trajectory, data['task'], t)
                state_hash = sha256(json.dumps({'state': state}, sort_keys=True).encode()).hexdigest()
                update_V(state, state_hash, V_target, outcome_reward, gamma, t)

    # Transform Q_target back to [0, 1]
    for key in V_target.keys():
        V_target[key]['vestimate'] = 0.5 * (V_target[key]['vestimate'] + 1)

    print_vestimate_histogram(V_target)

    # Split into train/test and save
    keys = list(V_target.keys())
    random.shuffle(keys)
    split_idx = int(len(keys) * train_split)
    train_keys, test_keys = keys[:split_idx], keys[split_idx:]
    
    train_data = [
        {'state': V_target[k]['state'], 
         'vestimate': V_target[k]['vestimate']}
        for k in train_keys
    ]
    test_data = [
        {'state': V_target[k]['state'], 
         'vestimate': V_target[k]['vestimate']}
        for k in test_keys
    ]

    # Convert lists of dictionaries to Arrow tables
    train_table = pa.Table.from_pylist(train_data)
    test_table = pa.Table.from_pylist(test_data)

    # Save the Arrow tables to Parquet files
    os.makedirs(outputdir, exist_ok=True)
    train_path = os.path.join(outputdir, 'train.parquet')
    test_path = os.path.join(outputdir, 'test.parquet')

    pq.write_table(train_table, train_path)
    pq.write_table(test_table, test_path)
    
def main():
    parser = argparse.ArgumentParser(description="Create training data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to dataproc config file"
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffling (default: 42)')

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    compute_vestimate(config["rolloutdirs"], config["outputdir"], config["gamma"], config["train_split"])

if __name__ == "__main__":
    main()
