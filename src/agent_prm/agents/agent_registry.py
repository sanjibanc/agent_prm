from typing import Callable, Dict, Any, Union
from agent_prm.agents.chat_agent import ChatAgent
from agent_prm.agents.hf_agent import HFAgent
from agent_prm.agents.hf_spaces_agent import HFSpaceAgent
from agent_prm.agents.sglang_server_agent import SGLangServerAgent
from agent_prm.critics.sglang_server_critic import SGLangServerCritic
from agent_prm.critics.random_critic import RandomCritic
from agent_prm.agents.best_of_n_agent import BestofNAgent
from agent_prm.agents.mixture_agent import MixtureAgent
from agent_prm.agents.switch_agent import SwitchAgent

def initialize_agent(
    agent_config: Dict[str, Any],
    parse_reason_action_fn: Callable,
    verbose: int = 0,
    debug: bool = False):
    """
    Initialize and return an agent based on the provided configuration.

    Args:
        agent_config (Dict[str, Any]): Configuration dictionary specifying the agent type and parameters.
        parse_reason_action_fn (Callable): Function to parse reason-action output.
        verbose (int, optional): Verbosity level. Defaults to 0.
        debug (bool, optional): Debugging flag. Defaults to False.

    Returns:
        The initialized agent instance.

    Raises:
        ValueError: If an unsupported agent type is specified.
    """
    
    agent_type = agent_config["type"]

    if agent_type == "chat":
        generate_fn = None
        if agent_config['model_type'] == "openai":
            from agent_prm.utils.openai import generate_from_openai_completion
            generate_fn = generate_from_openai_completion
        elif agent_config['model_type'] == "gemini":
            from agent_prm.utils.gemini import generate_from_gemini
            generate_fn = generate_from_gemini
        elif agent_config['model_type'] == "anthropic":
            from agent_prm.utils.anthropic import generate_from_anthropic
            generate_fn = generate_from_anthropic

        return ChatAgent(model_id=agent_config["model_id"],
                         prompt_template_file=agent_config["prompt_template_file"],
                         verbose=verbose,
                         debug=debug,
                         generate_fn=generate_fn,
                         parse_reason_action_fn=parse_reason_action_fn)
    elif agent_type == "hf":
        return HFAgent(model_id=agent_config["model_id"],
                       prompt_template_file=agent_config["prompt_template_file"],
                       verbose=verbose,
                       debug=debug,
                       parse_reason_action_fn=parse_reason_action_fn,
                       max_length=6000)
    elif agent_type == "hf_space":
        return HFSpaceAgent(model_id=agent_config["model_id"],
                            space_id=agent_config["space_id"],
                            prompt_template_file=agent_config["prompt_template_file"],
                            verbose=verbose,
                            debug=debug,
                            parse_reason_action_fn=parse_reason_action_fn,
                            max_length=6000)
    elif agent_type == "sglang_server":
        return SGLangServerAgent(model_id=agent_config["model_id"],
                                 server_url=agent_config["server_url"],
                                 prompt_template_file=agent_config["prompt_template_file"],
                                 verbose=verbose,
                                 debug=debug,
                                 parse_reason_action_fn=parse_reason_action_fn,
                                 temperature=agent_config["temperature"],
                                 batch_limit=agent_config["batch_limit"])
    elif agent_type == "best_of_n":
        generator = initialize_agent(agent_config=agent_config["generator"], parse_reason_action_fn=parse_reason_action_fn, verbose=verbose, debug=debug)        
        critic = initialize_critic(critic_config=agent_config["critic"], verbose=verbose, debug=debug)
        return BestofNAgent(generator=generator, 
                            critic=critic, 
                            num_generations=agent_config["num_generations"], 
                            verbose=verbose, 
                            debug=debug)
    elif agent_type == "mixture":
        agents = []
        for mixture_agent_config in agent_config["mixture_agents"]:
            agents.append(initialize_agent(agent_config=mixture_agent_config, parse_reason_action_fn=parse_reason_action_fn, verbose=verbose, debug=debug))
        return MixtureAgent(agents=agents, is_low_var=agent_config["is_low_var"], verbose=verbose, debug=debug)
    elif agent_type == "switch":
        agent_rollout = initialize_agent(agent_config=agent_config["agent_rollout"], parse_reason_action_fn=parse_reason_action_fn, verbose=verbose, debug=debug)        
        agent_switch = initialize_agent(agent_config=agent_config["agent_switch"], parse_reason_action_fn=parse_reason_action_fn, verbose=verbose, debug=debug)        
        return SwitchAgent(agent_rollout=agent_rollout, 
                           agent_switch=agent_switch, 
                           switch_time_horizon=agent_config["switch_time_horizon"], 
                           verbose=verbose, 
                           debug=debug)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    
def initialize_critic(
    critic_config: Dict[str, Any], verbose: int = 0, debug: bool = False
):
    """
    Initialize and return a critic based on the provided configuration.

    Args:
        critic_config (Dict[str, Any]): Configuration dictionary specifying the critic type and parameters.
        verbose (int, optional): Verbosity level. Defaults to 0.
        debug (bool, optional): Debugging flag. Defaults to False.

    Returns:
        Union[SGLangServerCritic, RandomCritic]: The initialized critic instance.

    Raises:
        ValueError: If an unsupported critic type is specified.
    """
    critic_type = critic_config["type"]
    if critic_type == "sglang_server":
        return SGLangServerCritic(model_id=critic_config["model_id"],
                                  server_url=critic_config["server_url"],
                                  prompt_template_file=critic_config["prompt_template_file"],
                                  verbose=verbose,
                                  debug=debug,
                                  batch_limit=critic_config["batch_limit"])
    elif critic_type == "random":
        return RandomCritic()      
    else:
        raise ValueError(f"Unsupported critic type: {critic_type}")
    
        
    
