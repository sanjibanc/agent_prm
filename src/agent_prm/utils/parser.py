import re
import json
from typing import Tuple, Optional, Dict, Any, List

def parse_reason_and_action_alfworld(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for ALFWorld environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action. 
    """
    reason_pattern = r"REASON:\s*(.*?)\s*ACTION:"
    action_pattern = r"ACTION:\s*([^\n]+)"

    reason_match = re.search(reason_pattern, text, re.DOTALL)
    action_match = re.search(action_pattern, text)

    reason = reason_match.group(1).strip() if reason_match else text
    action = action_match.group(1).strip() if action_match else ""

    #Clean up action to move to lower case and remove any random characters
    action = action.lower()
    action = re.sub(r'[^a-z0-9 /]', '', action)

    return reason, action

def parse_reason_and_action_webshop(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for WebShop environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action.
    """
    reason_pattern = r"REASON:\s*(.*?)\s*ACTION:"
    action_pattern = r"ACTION:\s*([^\n]+)"

    reason_match = re.search(reason_pattern, text, re.DOTALL)
    action_match = re.search(action_pattern, text)

    reason = reason_match.group(1).strip() if reason_match else text
    action = action_match.group(1).strip() if action_match else ""

    return reason, action

def parse_reason_and_action_intercode(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for Intercode environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action.
    """
    text = text.replace("\\n", "\n")
    reason_pattern = r"REASON:\s*(.*?)\s*ACTION:"
    
    # Two patterns for action: one with Markdown formatting and one without
    action_pattern_with_markdown = r"ACTION:\s*```(?:[a-zA-Z]*)?\s*(.*?)\s*```"
    # Updated pattern to capture only the first line of the action
    action_pattern_without_markdown = r"ACTION:\s*([^\n]+)"

    reason_match = re.search(reason_pattern, text, re.DOTALL)
    
    # Try matching the action with Markdown first
    action_match = re.search(action_pattern_with_markdown, text, re.DOTALL)
    
    # If no Markdown match, try the simpler non-Markdown pattern
    if not action_match:
        action_match = re.search(action_pattern_without_markdown, text, re.DOTALL)
    
    reason = reason_match.group(1).strip() if reason_match else text
    action = action_match.group(1).strip() if action_match else ""

    return reason, action

def parse_reason_and_action_intercode_sql(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for Intercode SQL environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and the cleaned action.
    """    
    reason, action = parse_reason_and_action_intercode(text)
    action = action.split('\n', 1)[0]
    # Further strip everything after the first semicolon (if any) but keep the semicolon
    if ';' in action:
        action = action.split(';', 1)[0] + ';'
    return reason, action

def preprocess_json_string(response: str) -> str:
    """
    Preprocesses a JSON string by escaping characters and normalizing spaces.

    Args:
        response: The raw JSON string.

    Returns:
        The preprocessed JSON string.
    """
    # Escape problematic characters
    response = response.replace("\\", "\\\\")  # escape backslashes
    response = re.sub(r'\s+', ' ', response)  # normalize spaces
    response = re.sub(r',\s*}', '}', response)  # remove trailing commas before closing braces
    response = re.sub(r',\s*]', ']', response)  # remove trailing commas before closing brackets
    corrected_response = re.sub(r'"\s*"trajectory"', '", "trajectory"', response)
    return corrected_response

def parse_corrected_trajectory(response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses the corrected trajectory from the given response text, extracting the JSON object from 
    within markdown-style code blocks.

    Args:
        response: The text containing the corrected trajectory.

    Returns:
        The parsed trajectory as a list of dictionaries, or None if the parsing fails.
    """
    try:
        json_str_match = re.search(r'```json\n(\{.*?\})\n```', response, re.DOTALL)

        if json_str_match is None:
            raise ValueError("No JSON content found")
        json_str = json_str_match.group(1)

        json_str = preprocess_json_string(json_str)
        parsed_dict = json.loads(json_str)
        corrected_trajectory = parsed_dict['trajectory']

        if not isinstance(corrected_trajectory, list) or not all(isinstance(item, dict) for item in corrected_trajectory):
            raise ValueError("corrected_trajectory is not a list of dicts")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        corrected_trajectory = None
        print(f"An error occurred: {e}")
    
    return corrected_trajectory

def preprocess_json_string(response: str) -> str:
    """
    Preprocesses a JSON-like string by escaping backslashes, normalizing spaces, and removing trailing 
    commas before closing braces or brackets.

    Args:
        response: The raw JSON string to be preprocessed.

    Returns:
        A cleaned JSON string with normalized formatting.
    """
    response = response.replace('\\"', "'") # Replace escaped double quotes with single quotes
    response = response.replace("\\", "\\\\")  # escape backslashes
    response = re.sub(r'\s+', ' ', response)  # normalize spaces
    response = re.sub(r',\s*}', '}', response)  # remove trailing commas before closing braces
    response = re.sub(r',\s*]', ']', response)  # remove trailing commas before closing brackets
    return response

def parse_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Parses a JSON object from the response, extracting it from markdown-style code blocks.

    Args:
        response: The text containing the JSON object.

    Returns:
        The parsed JSON object as a dictionary, or None if the parsing fails.
    """
    # Use regex to capture the JSON content between ```json and ```
    json_string = re.search(r'```json\n([\s\S]*?)\n```', response)
    if json_string:
        json_data = json_string.group(1)  # Extract the JSON string
        json_data = preprocess_json_string(json_data)  # Preprocess to fix common JSON issues
        try:
            # Parse the JSON string into a Python object
            parsed_data = json.loads(json_data)
            return parsed_data
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        print("No JSON data found")
        return None

def extract_task_from_observation_alfworld(observation: str) -> str:
    """
    Extracts the task description from the observation in ALFWorld.

    Args:
        observation: The observation text.

    Returns:
        The extracted task description.
    """
    match = re.search(r"(?<=Your task is to: ).*", observation)
    task = match.group() if match else ""
    return task

def substitute_placeholders(config: Any, template: str, value: str) -> Any:
    """
    Recursively substitutes placeholders in a configuration object with the provided value.

    Args:
        config: The configuration object
        template: The placeholder string to replace.
        value: The value to substitute in place of the placeholder.

    Returns:
        The configuration object with placeholders replaced.
    """
    if isinstance(config, dict):
        for key, sub_config in config.items():
            config[key] = substitute_placeholders(sub_config, template, value)
    elif isinstance(config, list):
        config = [substitute_placeholders(item, template, value) for item in config]
    elif isinstance(config, str):
        config = config.replace(template, value)
    return config

def parse_corrected_reason_and_action_alfworld(text: str) -> Tuple[str, str]:
    """
    Parses the corrected reason and action from the given text in the ALFWorld format.

    Args:
        text: The text containing the corrected reason and action.

    Returns:
        A tuple with the corrected reason and action. The action is cleaned by converting to lowercase and
        removing any non-alphanumeric characters.
    """
    reason_pattern = r"CORRECTED_REASON:\s*(.*?)\s*CORRECTED_ACTION:"
    action_pattern = r"CORRECTED_ACTION:\s*([^\n]+)"

    reason_match = re.search(reason_pattern, text, re.DOTALL)
    action_match = re.search(action_pattern, text)

    reason = reason_match.group(1).strip() if reason_match else text
    action = action_match.group(1).strip() if action_match else ""

    #Clean up action to move to lower case and remove any random characters
    action = action.lower()
    action = re.sub(r'[^a-z0-9 /]', '', action)

    return reason, action