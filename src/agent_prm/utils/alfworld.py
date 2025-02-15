import re
from typing import Optional

def parse_gamefile(env) -> Optional[str]:
    """
    Parses the game file from the environment and extracts the file name after a specific marker.

    Args:
        env: The environment object, expected to have a batch of environments with a `gamefile` or `_game_file` attribute.

    Returns:
        The extracted part of the game file after the "json_2.1.1/" marker as a string, or None if the marker is not found.
    """
    pddl_env = env.batch_env.envs[0]
    if hasattr(pddl_env, 'gamefile'):
        gamefile = pddl_env.gamefile
    else:
        gamefile = pddl_env._game_file

    marker = "json_2.1.1/"
    index = gamefile.find(marker)
    if index != -1:
        return gamefile[index + len(marker):]
    else:
        return None

def parse_gamefile_string(gamefile) -> Optional[str]:
    marker = "json_2.1.1/"
    index = gamefile.find(marker)
    if index != -1:
        return gamefile[index + len(marker):]
    else:
        return None   

def extract_task_from_observation(observation: str) -> str:
    """
    Extracts the task description from the observation string by searching for the "Your task is to:" phrase.

    Args:
        observation: A string containing the observation, which includes the task description.

    Returns:
        The extracted task description as a string, or an empty string if the task is not found.
    """
    match = re.search(r"(?<=\n\nYour task is to: ).*", observation)
    task = match.group() if match else ""
    return task