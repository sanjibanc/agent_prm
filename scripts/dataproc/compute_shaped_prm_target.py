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
from agent_prm.agents.agent_registry import initialize_critic
import copy

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

def update_Q(state, reason_action, state_hash, Q_target, q_target):
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
    
    updated_qestimate = (current_qestimate * current_count + q_target) / (current_count + 1)
    Q_target[state_hash]['qestimate'] = updated_qestimate
    Q_target[state_hash]['count'] = current_count + 1

def compute_critic_values(all_states, critic):
    queries = [
                {
                    "task": state["task"],
                    "observation": state["observation"],
                    "candidate_actions": state["candidate_actions"],
                    "observation_action_history": state["history"],
                }
                for state in list(all_states.values())
            ]
    scores = critic.score_state_batch(queries=queries)
    
    critic_values = {}
    for (state_hash, _), score in zip(all_states.items(), scores):
        critic_values[state_hash] = score

    return critic_values


def transform_regression_to_soft_classification(Q_target, c=1):
    # Extract 'qestimate' values from Q_target
    y = np.array([val['qestimate'] for val in Q_target.values()])
    
    # Compute median for threshold T
    T = np.mean(y)
    
    # Compute IQR for steepness k
    q75, q25 = np.percentile(y, [75, 25])
    iqr = q75 - q25
    k = c / iqr if iqr != 0 else 1  # Avoid division by zero
    
    # Compute soft labels using sigmoid function
    p = 1 / (1 + np.exp(-k * (y - T)))
    
    # Update Q_target in place with new 'qestimate' values
    Q_target_adjusted = copy.deepcopy(Q_target)
    for i, key in enumerate(Q_target):
        Q_target_adjusted[key]['qestimate'] = p[i]
    return Q_target_adjusted


def compute_shaped_qestimate(rolloutdirs, critic_config, outputdir, gamma, train_split, shaping_constant):
    critic = initialize_critic(critic_config=critic_config, verbose=1)
    
    state_action_infos = []
    all_states = {}
    counter = 0
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
            
            # counter +=1
            # if counter >= 10:
            #     break
            
            trajectory = data['trajectory']
            outcome_reward = 2 * trajectory[-1]['score'] - 1  # transform from [-1, 1]
            for t in range(len(trajectory) - 1, -1, -1):
                state, reason_action = extract_state_reason_action(trajectory, data['task'], t)
                
                state_action_info = {
                    'state': state,
                    'reason_action': reason_action,
                    'outcome_reward': outcome_reward,
                    't': t,
                }
                if t == len(trajectory) - 1:
                    state_action_info['next_state'] = None
                else:
                    state_action_info['next_state'], _ = extract_state_reason_action(trajectory, data['task'], t+1)
                state_action_infos.append(state_action_info)
                
                state_hash = sha256(json.dumps({'state': state}, sort_keys=True).encode()).hexdigest()
                all_states[state_hash] = state
                
    # Compute critic for all unique states
    critic_values = compute_critic_values(all_states, critic)
                    
    Q_target = {}
    for state_action_info in state_action_infos:
        state, reason_action, outcome_reward, t = state_action_info['state'], state_action_info['reason_action'], state_action_info['outcome_reward'], state_action_info['t']
        state_action_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
        q_target = (gamma ** t) * outcome_reward
        
        if state_action_info['next_state'] is not None:
            state_hash = sha256(json.dumps({'state': state}, sort_keys=True).encode()).hexdigest()
            next_state_hash = sha256(json.dumps({'state': state_action_info['next_state']}, sort_keys=True).encode()).hexdigest()
            q_target += gamma*critic_values[next_state_hash] - critic_values[state_hash]
        
        update_Q(state, reason_action, state_action_hash, Q_target, q_target)
    
    print_qestimate_histogram(Q_target)
    # Transform Q_target to [0, 1]
    Q_target_normalized = transform_regression_to_soft_classification(Q_target, c=shaping_constant)

    print_qestimate_histogram(Q_target_normalized)

    # Split into train/test and save
    keys = list(Q_target_normalized.keys())
    random.shuffle(keys)
    split_idx = int(len(keys) * train_split)
    train_keys, test_keys = keys[:split_idx], keys[split_idx:]
    
    train_data = [
        {'state': Q_target_normalized[k]['state'], 
         'reason_action': Q_target_normalized[k]['reason_action'], 
         'qestimate': Q_target_normalized[k]['qestimate']}
        for k in train_keys
    ]
    test_data = [
        {'state': Q_target_normalized[k]['state'], 
         'reason_action': Q_target_normalized[k]['reason_action'], 
         'qestimate': Q_target_normalized[k]['qestimate']}
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
    
    #### Save raw values for posterity
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
    outputdir_raw = outputdir+"/raw"
    os.makedirs(outputdir_raw, exist_ok=True)
    train_path = os.path.join(outputdir_raw, 'train.parquet')
    test_path = os.path.join(outputdir_raw, 'test.parquet')

    pq.write_table(train_table, train_path)
    pq.write_table(test_table, test_path)
    
    

    # # Save 10% of the data to JSON
    # train_json_sample = random.sample(train_data, int(len(train_data) * 0.1))
    # train_json_path = os.path.join(outputdir, 'train.json')
    # with open(train_json_path, 'w') as train_json_file:
    #     json.dump(train_json_sample, train_json_file, indent=4)
    # test_json_sample = random.sample(test_data, int(len(test_data) * 0.1))
    # test_json_path = os.path.join(outputdir, 'test.json')
    # with open(test_json_path, 'w') as test_json_file:
    #     json.dump(test_json_sample, test_json_file, indent=4)


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
    compute_shaped_qestimate(config["rolloutdirs"], config["critic"], config["outputdir"], config["gamma"], config["train_split"], config["shaping_constant"])

if __name__ == "__main__":
    main()
