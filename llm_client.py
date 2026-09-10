import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

USE_LOCAL = False

if USE_LOCAL:
    _client = OpenAI(base_url="http://172.22.144.1:11434/v1", api_key="ollama")
    DEFAULT_MODEL = "llama3.2:3b"
else:
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
    DEFAULT_MODEL = "mistralai/mistral-nemotron"

def ask_llm(messages: list[dict], model: str = DEFAULT_MODEL, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                top_p=0.7,
                max_tokens=1024,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries:
                return f"(Model temporarily unavailable after {retries+1} attempts: {e})"
            time.sleep(2)


# def ask_llm(messages: list[dict], model: str = "mistralai/mistral-nemotron") -> str:
#     response = _client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.6,
#         top_p=0.7,
#         max_tokens=4096,
#         stream=False
#     )
#     return response.choices[0].message.content