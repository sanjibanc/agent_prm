import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple
import anthropic

def generate_from_anthropic(
    messages: List[dict[str, str]],
    model: str,
) -> Tuple[str, float]:
    """
    Generates a completion from Anthropic model.

    Args:
        messages (list[dict[str, str]]): The messages to send to the model.
        model (str): The model to use for generating the completion.

    Returns:
        Tuple[str, float]: The content of the model's response and the cost of the operation.
    """
    load_dotenv(find_dotenv())
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response_obj = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=messages
    )

    response = response_obj.content[0].text
    cost = 0.0 #TODO implement cost for anthropic
    return response, cost