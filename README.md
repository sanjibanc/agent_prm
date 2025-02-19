# <img src="assets/icon.png" alt="Icon" width="50"/> Process Reward Models for LLM Agents: Practical Framework and Directions

Paper link: [https://arxiv.org/pdf/2502.10325](https://arxiv.org/pdf/2502.10325)

## Installation

### Create Conda environment

To set up the project, clone the repository and create a Conda environment:

```bash
cd agent_prm
conda env create -f environment.yml
conda activate agent_prm
pip install -e .
```

### Optional: Set up OpenAI / Gemini / Anthropic environment keys 
Ensure you have a `.env` file with the requisite keys:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORGANIZATION=your_openai_organization_id
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Set up external dependencies

We build on [OpenInstruct](https://github.com/allenai/open-instruct) for training, with some minor compatibility fixes so it needs to be installed locally. 
```bash
# Clone and install Open-Instruct
git clone --branch fix_vllm https://github.com/sanjibanc/open-instruct.git
cd open-instruct
pip install -e .
cd ..
```

We use [SGLang](https://github.com/sgl-project/sglang) for fast inference, with some  minor compatibility fixes with LLama so it needs to be installed locally. 
```bash
# Clone and install SGLang
git clone --branch new_llama_model https://github.com/sanjibanc/sglang.git
cd sglang
pip install -e .
cd ..
```
To use slgang server, [got to SGlang instructions](#sglang-instructions)

To set up external environments like AlfWorld, [go to external environment instructions](#external-environment-instructions).

## Agent PRM Training

Agent PRM iterates over 3 stages:
1. Rollout policy and compute PRM targets
2. Train PRM
3. Train policy via RL

Stage 2 and 3 are similar to standard RLHF operations, with stage 1 being the agent specific step. 

### Initialize policy with SFT

We collect SFT training data from our prior work [LEAP](https://github.com/sanjibanc/leap_llm) and train a policy via SFT
```bash
bash bash/train-sft-llama3.2-3B.sh
```
For simplicity we provide the model here [rl-llm-agent/Llama-3.2-3B-Instruct-sft-alfworld-iter0](rl-llm-agent/Llama-3.2-3B-Instruct-sft-alfworld-iter0)

### Stage 1: Rollout and Compute Target

We rollout in a batched fashion, and recommend using the SGLangServerAgent for fast inference. See [sglang instructions](#sglang-instructions) to setup the SGLang server, then run the following script to collect rollouts

```bash
python scripts/dataproc/rollout_alfworld.py --config configs/rollout_alfworld.yaml
```

Once you have the rollouts, set the rollout path in `configs/compute_prm_target` and run
```bash
python scripts/dataproc/compute_prm_target.py --config configs/compute_prm_target.yaml
```

This should create a train and test file to train the PRM

### Stage 2: Training the PRM

To train the PRM, run the script that calls open instruct
```bash
bash bash/train-rm-llama3.2-3B.sh
```

Upload the best checkpoint to HF for convenience
```bash
python scripts/utils/upload_model_to_hf.py --input_model <path/to/checkpoint>  --output_model <hugging face model path> --accelerate
```

### Stage 3: Training the Policy via RL

To train the policy via OnlineDPO to optimize the PRM, run the following script
```bash
bash bash/online-dpo-llama3.2-3B.sh
```

Upload the best checkpoint to HF for convenience

Repeat stages 1 to 3. 

## Agent PRM Inference

Configure the agents you want to evaluate in `configs/eval_alfworld.yaml` and run the following script:
```bash
python scripts/eval/eval_alfworld.py --config configs/evaluate_alfworld.yaml
```
It will create a folder in `data/eval/alfworld/` with the current datetime where logs and summary.csv will be saved.

For fast inference, use a SGLang server agent and host the policy in a SGLang server. 

To evaluate a Best-of-N policy, host both the policy and the PRM in SGLang, and run the script with best_of_n agent.

## Ablations and Extensions

### Inverse PRM

Stage 1: Given expert demonstrations and policy rollouts, compute inverse PRM target

```bash
python scripts/dataproc/compute_inverse_prm_target.py --config configs/compute_inverse_prm_target.yaml
```

Stage 2: Train PRM

```bash
bash bash/train-inverse-prm-llama3.2-3B.sh
```

Stage 3: Train generator as in agent prm

### Relative Loss

To train the PRM using a relative loss, change the target computation to be a preference dataset

```bash
python scripts/dataproc/compute_prm_preference_target.py --config configs/compute_prm_preference_target.yaml
```

To train the PRM using preference data, use the script

```bash
bash  bash/train-rm-pref-llama3.2-3B.sh
```

### Steered Exploration

To train the policy using a steered exploration prompt `prompts/alfworld/alfworld_exploration_template.j2`, run the following script

```bash
python scripts/dataproc/compute_value_target.py --config configs/<path to value target.yaml>
bash bash/train-value-model-llama3.2-3B.sh
```

### Process Reward Shaping

Given a reference policy, collect rollouts, compute value targets and train a value estimate
```bash
bash bash/online-dpo-exploration-llama3.2-3B.sh
```

Use the value function to compute shaped PRM targets. This requires running the value function as a critic in a SGLang server
```bash
python scripts/dataproc/compute_shaped_prm_target.py --config configs/compute_shaped_prm_target.yaml
```

Train the shaped PRM
```bash
bash bash/train-shaped-rm-llama3.2-3B.sh
```

Train the policy via online DPO
```bash
bash bash/online-dpo-shaped-prm-llama3.2-3B.sh
```

## SGLang instructions

To use SGLang for inference, grab a node from the same network as your inference scripts so they can communicate over the network. 

SGLang has some compatibility issues with agent_prm conda environment, so we recommend using the sglang environment
```bash
conda env create -f sglang_environment.yml
conda activate sglang
```
To host a model, run
```bash
python -m sglang.launch_server --model-path <model_name> --port <port_number, e.g. 30000>
```

When doing inference for Best-of-N with a PRM, you might want to grab two such nodes, one for the generator, and one for the verifier and assign them two different ports 3000 and 30010.

## External environment instructions

### Setup AlfWorld
Clone AlfWorld from [AlfWorld github repository](https://github.com/alfworld/alfworld). Follow the instructions in its README to get the game files.

Create an env_assets folder and copy over data to `env_assets/alfworld`. Set the following environment variable:
```bash
export ALFWORLD_DATA=</path/to/env_assets/alfworld>
```

## Contact

This project is is actively being developed. For any questions or issues, please contact us at sanjibanc@cornell.edu.
