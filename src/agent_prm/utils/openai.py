import os
import openai
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple


input_token_cost_usd_by_model = {
    "gpt-4o": 5/1e6,
    "gpt-4o-mini": 0.15/1e6,
    "gpt-4-1106-preview": 0.01 / 1000,
    "gpt-4": 0.03 / 1000,
    "gpt-4-32k": 0.06 / 1000,
    "gpt-3.5-turbo": 0.001 / 1000,
    "gpt-3.5-turbo-instruct": 0.0015 / 1000,
    "gpt-3.5-turbo-16k": 0.003 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

output_token_cost_usd_by_model = {
    "gpt-4o": 15/1e6,
    "gpt-4o-mini": 0.6/1e6,
    "gpt-4-1106-preview": 0.03 / 1000,
    "gpt-4": 0.06 / 1000,
    "gpt-4-32k": 0.12 / 1000,
    "gpt-3.5-turbo": 0.002 / 1000,
    "gpt-3.5-turbo-instruct": 0.002 / 1000,
    "gpt-3.5-turbo-16k": 0.004 / 1000,
    "babbage-002": 0.0016 / 1000,
    "davinci-002": 0.012 / 1000,
    "ada-v2": 0.0001 / 1000,
}

def generate_from_openai_completion(
    messages: List[dict[str, str]],
    model: str,
    temperature: float = 0.3,
    top_p: float = 1.0,
    n: int = 1,
) -> Tuple[str, float]:
    """
    Generates a completion from OpenAI's GPT model.

    Args:
        messages (list[dict[str, str]]): The messages to send to the model.
        model (str): The model to use for generating the completion.

    Returns:
        Tuple[str, float]: The content of the model's response and the cost of the operation.
    """
    load_dotenv(find_dotenv())
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORGANIZATION", ""),
    )
    chat_completion = client.chat.completions.create(
        messages=messages, model=model, temperature=temperature, top_p=top_p, n=n
    )
    response = chat_completion.choices[0].message.content
    cost = (
        chat_completion.usage.prompt_tokens * input_token_cost_usd_by_model[model]
        + chat_completion.usage.completion_tokens
        * output_token_cost_usd_by_model[model]
    )

    return response, cost