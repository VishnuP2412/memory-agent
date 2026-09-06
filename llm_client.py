import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

def ask_llm(messages: list[dict], model: str = "mistralai/mistral-nemotron") -> str:
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=False
    )
    return response.choices[0].message.content