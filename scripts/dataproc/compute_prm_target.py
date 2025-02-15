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
from multiprocessing import Pool, cpu_count
from functools import partial

def print_qestimate_histogram(Q_target, bins=10):
    qestimates = [entry['qestimate'] for entry in Q_target.values()]
    counts, bin_edges = np.histogram(qestimates, bins=bins)
    
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

def update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, t):
    if state_hash not in Q_target:
        Q_target[state_hash] = {
            'state': state,
            'reason_action': reason_action,
            'qestimate': 0,
            'count': 0
        }
    current_entry = Q_target[state_hash]
    current_qestimate = current_entry['qestimate']
    current_count = current_entry['count']
    
    updated_qestimate = (current_qestimate * current_count + (gamma ** t) * outcome_reward) / (current_count + 1)
    Q_target[state_hash]['qestimate'] = updated_qestimate
    Q_target[state_hash]['count'] = current_count + 1

def process_file(file_path, gamma):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if 'trajectory' not in data:
            return {}

        Q_target = {}
        trajectory = data['trajectory']
        outcome_reward = 2 * trajectory[-1]['score'] - 1  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = extract_state_reason_action(trajectory, data['task'], t)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, t)
        return Q_target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

# Function to merge results from multiple processes
def merge_results(results):
    merged_Q_target = {}
    for Q_target in results:
        for key, value in Q_target.items():
            if key not in merged_Q_target:
                merged_Q_target[key] = value
            else:
                # Merge results for duplicate keys
                current_entry = merged_Q_target[key]
                new_qestimate = (current_entry['qestimate'] * current_entry['count'] + 
                                 value['qestimate'] * value['count']) / (current_entry['count'] + value['count'])
                merged_Q_target[key]['qestimate'] = new_qestimate
                merged_Q_target[key]['count'] += value['count']
    return merged_Q_target

# Main function using multiprocessing
def compute_prm_target(files, outputdir, gamma, train_split):
    Q_target = {}

    # Automatically detect the number of CPUs
    num_cpus = cpu_count()

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=num_cpus) as pool:
        process_func = partial(process_file, gamma=gamma)
        results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing files"))
    
    # Merge results from all processes
    Q_target = merge_results(results)

    # Transform Q_target back to [0, 1]
    for key in Q_target.keys():
        Q_target[key]['qestimate'] = 0.5 * (Q_target[key]['qestimate'] + 1)

    print_qestimate_histogram(Q_target)

    # Split into train/test and save
    keys = list(Q_target.keys())
    random.shuffle(keys)
    split_idx = int(len(keys) * train_split)
    train_keys, test_keys = keys[:split_idx], keys[split_idx:]

    train_data = [
        {'state': Q_target[k]['state'], 
         'reason_action': Q_target[k]['reason_action'], 
         'qestimate': Q_target[k]['qestimate']}
        for k in train_keys
    ]
    test_data = [
        {'state': Q_target[k]['state'], 
         'reason_action': Q_target[k]['reason_action'], 
         'qestimate': Q_target[k]['qestimate']}
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

def compute_file_list(rolloutdirs, max_files_per_dir=None):
    files = []
    for rolloutdir in rolloutdirs:
        files_per_dir = []
        for file_name in tqdm(os.listdir(rolloutdir)):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(rolloutdir, file_name)
            files_per_dir.append(file_path)
            if (max_files_per_dir is not None) and (len(files_per_dir) >= max_files_per_dir):
                break
        files = files + files_per_dir
    return files
        

def main():
    parser = argparse.ArgumentParser(description="Create PRM target")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to dataproc config file"
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffling (default: 42)')

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    files = compute_file_list(config["rolloutdirs"], config["max_files_per_dir"])
    compute_prm_target(files, config["outputdir"], config["gamma"], config["train_split"])

if __name__ == "__main__":
    main()
