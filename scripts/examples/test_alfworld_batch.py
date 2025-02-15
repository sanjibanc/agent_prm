import time
import numpy as np
from tqdm import tqdm
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import alfworld.agents.modules.generic as generic

def test_step_speed(batch_sizes):
    """
    Test how the speed of environment steps varies with batch size.
    """
    # Load config and initialize environment type
    config = generic.load_config()

    results = []
    max_steps = 20  # Number of steps to test

    for batch_size in batch_sizes:
        # Initialize the environment with the given batch size
        env = AlfredTWEnv(config, train_eval='train')
        env = env.init_env(batch_size=batch_size)

        # Reset environment
        observations, info = env.reset()

        start_time = time.time()

        for _ in tqdm(range(max_steps), desc=f"Batch size: {batch_size}"):
            # Use a dummy action ("look") for all agents in the batch
            actions = ["look"] * batch_size
            observations, scores, dones, info = env.step(actions)

        elapsed_time = time.time() - start_time
        avg_step_time = elapsed_time / max_steps
        results.append((batch_size, avg_step_time))
        print(f"Batch size: {batch_size}, Avg step time: {avg_step_time:.4f} seconds")

    return results


if __name__ == "__main__":
    # Test different batch sizes
    batch_sizes = [1, 16, 64, 128]  # Example batch sizes
    results = test_step_speed(batch_sizes)

    # Print summary
    print("\nSummary:")
    for batch_size, avg_step_time in results:
        print(f"Batch size: {batch_size}, Avg step time: {avg_step_time:.4f} seconds")
