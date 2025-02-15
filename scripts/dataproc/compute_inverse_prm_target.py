import os
import json
import argparse
import yaml
import random
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.utils.parser import parse_reason_and_action_alfworld


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

def extract_sasa(files):
    sasas = [] #[(state, action, next_state_action)]
    for file_path in tqdm(files):                
        # Load JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if 'trajectory' not in data:
            continue
                    
        trajectory = data['trajectory']
                    
        state_reason_actions = []
        for t in range(len(trajectory)):
            state, reason_action = extract_state_reason_action(trajectory, data['task'], t)
            state_reason_actions.append({'state': state, 'reason_action': reason_action})
        
        for t in range(len(state_reason_actions)):
            if t < len(state_reason_actions) - 1:
                sasas.append({
                    "state": state_reason_actions[t]['state'],
                    "reason_action": state_reason_actions[t]['reason_action'],
                    "next_state": state_reason_actions[t+1]['state'],
                    "next_reason_action": state_reason_actions[t+1]['reason_action'],
                    "terminal": False
                })
            else:
                sasas.append({
                    "state": state_reason_actions[t]['state'],
                    "reason_action": state_reason_actions[t]['reason_action'],
                    "next_state": None,
                    "next_reason_action": None,
                    "terminal": True
                })                    
    return sasas
              
def overwrite_next_action(sasas, policy):
    queries = []
    for sasa in sasas:
        if sasa["terminal"]:
            continue
        queries.append({
                    "task": sasa["next_state"]["task"],
                    "observation": sasa["next_state"]["observation"],
                    "candidate_actions": sasa["next_state"]["candidate_actions"],
                    "observation_action_history": sasa["next_state"]["history"],
                })

    reason_actions_all_queries = policy.predict_reason_action_batch(queries=queries,num_responses=1)
    
    for (sasa, reason_actions_per_query) in zip(sasas, reason_actions_all_queries):
        if sasa["terminal"]:
            continue
        assert len(reason_actions_per_query) == 1
        reason_action = reason_actions_per_query[0]
        sasa["next_reason_action"] = reason_action
    
    return sasas
  

def balance_pos_neg_data(pos_rolloutdirs, neg_rolloutdirs, balance_pos_neg):
    pos_files_alldir = []
    for rolloutdir in pos_rolloutdirs:
        pos_files = []
        for file_name in os.listdir(rolloutdir):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(rolloutdir, file_name)
            pos_files.append(file_path)
        pos_files_alldir.append(pos_files)

    neg_files_alldir = []
    neg_files = []
    for rolloutdir in neg_rolloutdirs:
        for file_name in os.listdir(rolloutdir):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(rolloutdir, file_name)
            neg_files.append(file_path)
        neg_files_alldir.append(neg_files)

    if balance_pos_neg:
        total_pos = (min(len(lst) for lst in pos_files_alldir)) * len(pos_files_alldir)
        total_neg = (min(len(lst) for lst in neg_files_alldir)) * len(neg_files_alldir)
        target_size = min(total_pos, total_neg)
        target_pos_size = int(target_size/len(pos_files_alldir))
        target_neg_size = int(target_size/len(neg_files_alldir))
        
        for i, pos_files in enumerate(pos_files_alldir):
            pos_files_alldir[i] = random.sample(pos_files, target_pos_size)
        for i, neg_files in enumerate(neg_files_alldir):
            neg_files_alldir[i] = random.sample(neg_files, target_neg_size)

        
    
    pos_all_files = [item for sublist in pos_files_alldir for item in sublist]
    neg_all_files = [item for sublist in neg_files_alldir for item in sublist]

    return pos_all_files, neg_all_files
    

def compute_iqlearn_data(pos_rolloutdirs, neg_rolloutdirs, balance_pos_neg, policy_config, train_split, outputdir):
    policy = initialize_agent(agent_config=policy_config, 
                            parse_reason_action_fn=parse_reason_and_action_alfworld,
                            verbose=policy_config['verbose'])
    
    pos_all_files, neg_all_files = balance_pos_neg_data(pos_rolloutdirs, neg_rolloutdirs, balance_pos_neg)
    print(f"Positive files: {len(pos_all_files)} Negative files: {len(neg_all_files)}")
    
    pos_sasas = extract_sasa(pos_all_files)
    pos_sasas = overwrite_next_action(pos_sasas, policy)
    for pos_sasa in pos_sasas:
        pos_sasa['label']=1.0
    
    neg_sasas = extract_sasa(neg_all_files)
    for neg_sasa in neg_sasas:
        neg_sasa['label']=0.0
    
    combined_sasa = pos_sasas + neg_sasas
    random.shuffle(combined_sasa)
    
    # train, test split
    split_idx = int(len(combined_sasa) * train_split)
    train_sasas, test_sasas = combined_sasa[:split_idx], combined_sasa[split_idx:]
    
    # Convert lists of dictionaries to Arrow tables
    train_table = pa.Table.from_pylist(train_sasas)
    test_table = pa.Table.from_pylist(test_sasas)
    os.makedirs(outputdir, exist_ok=True)
    train_path = os.path.join(outputdir, 'train.parquet')
    test_path = os.path.join(outputdir, 'test.parquet')
    pq.write_table(train_table, train_path)
    pq.write_table(test_table, test_path)
    

def main():
    parser = argparse.ArgumentParser(description="Generate IQ learn data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to IQ learn config file"
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffling (default: 42)')

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    compute_iqlearn_data(config["pos_rolloutdirs"], 
                         config["neg_rolloutdirs"], 
                         config["balance_pos_neg"],
                         config["policy"], 
                         config["train_split"], 
                         config["outputdir"])

if __name__ == "__main__":
    main()