import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple
import google.generativeai as genai

def generate_from_gemini(
    messages: List[dict[str, str]],
    model: str,
) -> Tuple[str, float]:
    """
    Generates a completion from Gemini model.

    Args:
        messages (list[dict[str, str]]): The messages to send to the model. Assume there is only one message
        model (str): The model to use for generating the completion.

    Returns:
        Tuple[str, float]: The content of the model's response and the cost of the operation.
    """
    assert len(messages) == 1 # assume there is only one message
    load_dotenv(find_dotenv())
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = genai.GenerativeModel(model)
    chat = client.start_chat()
    response_obj = chat.send_message(messages[0]['content'])
    response = response_obj.text
    cost = 0.0 # TODO implement cost for Gemini model
    return response, cost