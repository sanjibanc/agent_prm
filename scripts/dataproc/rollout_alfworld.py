import os
import sys
import time
import json
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import alfworld.agents.modules.generic as generic

from agent_prm.agents.agent_registry import initialize_agent

from agent_prm.utils.alfworld import parse_gamefile_string, extract_task_from_observation
from agent_prm.utils.parser import parse_reason_and_action_alfworld

def parse_and_load_config():
    """
    Parse command-line arguments and load the appropriate config file.
    """
    parser = argparse.ArgumentParser(description="Evaluate agent on ALFWorld")
    parser.add_argument("--alfworld_config", type=str, default="configs/env_config/alfworld_config.yaml", help="Path to the Alfred base config file")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    # Update sys.argv for compatibility with generic.load_config
    sys.argv = [sys.argv[0], args.alfworld_config]

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def rollout_agent_env(env, agent):
    agent.reset()
    observations, info = env.reset()
    num_env_batch = len(observations)
    
    if len(env.batch_env.envs) == 1:
        max_actions = env.batch_env.envs[0].max_episode_steps
    else:
        max_actions = env.batch_env.envs[0].get_sync("max_episode_steps")

    env_datas = [
        {
            "gamefile": parse_gamefile_string(info['extra.gamefile'][b]),
            "task": extract_task_from_observation(observations[b]),
            "observation": [observations[b]],
            "candidate_actions": [info["admissible_commands"][b]],
            "reason": [],
            "action": [],
            "score": [],
            "agent_log": [],
            "done": False
        } for b in range(num_env_batch)
    ]
    
    for _ in tqdm(range(max_actions), desc=f"Actions", leave=False):
        try:
            queries = [
                {
                    "task": env_data["task"],
                    "observation": env_data["observation"][-1],
                    "candidate_actions": env_data["candidate_actions"][-1],
                    "observation_action_history": [{"observation": env_data["observation"][t], "action": env_data["observation"][t]} for t in range(0, len(env_data["observation"])-1)],
                }
                for env_data in env_datas
            ]
            
            reason_actions_all_queries = agent.predict_reason_action_batch(queries=queries,num_responses=1)
            agent_log_all_queries = agent.get_log()
        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None

        actions = []
        for reason_actions_per_query in reason_actions_all_queries:
            assert len(reason_actions_per_query) == 1
            reason_action = reason_actions_per_query[0]
            actions.append(reason_action['action'])

        observations, scores, dones, info = env.step(actions)

        for b in range(num_env_batch):
            if not env_datas[b]["done"]:
                reason_actions_per_query = reason_actions_all_queries[b]
                assert len(reason_actions_per_query) == 1
                reason_action = reason_actions_per_query[0]

                env_datas[b]["reason"].append(reason_action['reason'])
                env_datas[b]["action"].append(reason_action['action'])
                env_datas[b]["score"].append(scores[b])
                env_datas[b]["done"] = dones[b] 
                env_datas[b]["observation"].append(observations[b])
                env_datas[b]["candidate_actions"].append(info["admissible_commands"][b])
                if agent_log_all_queries is not None:
                    env_datas[b]["agent_log"].append(agent_log_all_queries[b])
                else:
                    env_datas[b]["agent_log"].append([])
            
        if all([env_data["done"] for env_data in env_datas]):
            break

    rollouts = []
    for env_data in env_datas:  
        trajectory = [
            {
                "observation": env_data["observation"][t],
                "candidate_actions": env_data["candidate_actions"][t],
                "reason": env_data["reason"][t],
                "action": env_data["action"][t],
                "score": env_data["score"][t],
                "agent_log": env_data["agent_log"][t],
            }
            for t in range(0, len(env_data["action"]))
        ]
        rollouts.append({"gamefile": env_data["gamefile"], "trajectory": trajectory, "task": env_data["task"]})

    # Append to the summary file
    summaries = [{
        "gamefile": env_data["gamefile"],
        "model_id": agent.name(),
        "num_actions": len(env_data["action"]),
        "score": env_data["score"][-1]
    } for env_data in env_datas]
    return rollouts, summaries

def main():
    config = parse_and_load_config()
    alfworld_config = generic.load_config()

    dstdir = os.path.join(config['logdir'], time.strftime('%Y%m%d-%H%M%S')) if not config["exact_path"] else config["logdir"]
    os.makedirs(dstdir, exist_ok=True)
    with open(os.path.join(dstdir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    log_counter = 0
    start_time = time.time()  # Track the start time

    for agent_config in config["agents"]: # loop over agents
        agent = initialize_agent(agent_config=agent_config, 
                                 parse_reason_action_fn=parse_reason_and_action_alfworld, 
                                 verbose=config["verbose"], 
                                 debug=config["debug"])
        logdir = os.path.join(dstdir, agent.name()) if not config['exact_path'] else config['logdir']
        os.makedirs(logdir, exist_ok=True)

        env_generator = AlfredTWEnv(alfworld_config, train_eval=config["eval_set"])
        env = env_generator.init_env(batch_size=config['env_batch_size'])
        for env_seed in range(config["num_env_batch"]):
            for _ in range(config["num_rollouts_per_env"]):
                env.seed(env_seed+config['env_seed'])
                rollouts, summaries = rollout_agent_env(env, agent)

                if (rollouts is None) or (summaries is None):
                    continue
            
                # save rollouts
                for rollout in rollouts:
                    log_file_path = os.path.join(logdir, f"{log_counter}.json")
                    log_counter = log_counter + 1
                    with open(log_file_path, "w") as log_file:
                        json.dump(rollout, log_file, indent=4)
                        
                # save (update) summary
                summary_file_path = os.path.join(dstdir, "summary.csv")
                if os.path.exists(summary_file_path):
                    df_summary = pd.read_csv(summary_file_path)
                else:
                    df_summary = pd.DataFrame()

                df_summary = pd.concat(
                    [df_summary, pd.DataFrame(summaries)], ignore_index=True
                )
                df_summary.to_csv(summary_file_path, index=False)
            
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                completed_rollouts = len(df_summary)
                total_rollouts = config["num_env_batch"] * config['env_batch_size'] * config["num_rollouts_per_env"]
                remaining_time = (elapsed_time / completed_rollouts) * (total_rollouts - completed_rollouts) if completed_rollouts > 0 else float('inf')
                print(f"Num rollouts completed: {completed_rollouts} Elapsed time: {elapsed_time:.2f}s Estimated time left: {remaining_time:.2f}s\n")
            

if __name__ == "__main__":
    main()
