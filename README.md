# Memory Agent

A simple Python command-line assistant with persistent semantic memory.

The agent stores user messages and assistant replies in a local ChromaDB
collection. Before each response, it retrieves relevant past memories and
adds them to the prompt sent to the NVIDIA-hosted language model.

## Features

- Interactive terminal chat
- Persistent local memory in `chroma_db/`
- Semantic search with `all-MiniLM-L6-v2`
- NVIDIA API access through the OpenAI-compatible client
- Recency and memory-compaction helpers in `retention.py`

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install chromadb sentence-transformers openai python-dotenv
```

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

## Run

```bash
python chat.py
```

Type `exit` to end the conversation. The local `chroma_db/` directory stores
the conversation memories between runs.

## Project Structure

| File              | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `chat.py`         | Command-line chat loop and context retrieval |
| `memory_store.py` | Embeddings and ChromaDB persistence          |
| `llm_client.py`   | NVIDIA API client                            |
| `retention.py`    | Recency scoring and compaction helpers       |

## License

This project does not currently include a license.
