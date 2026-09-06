# Memory Agent

A small Python command-line assistant with persistent semantic memory. It
implements a retrieval-augmented generation (RAG) loop:

1. Embed the user's message with `all-MiniLM-L6-v2`.
2. Store user memories in a local ChromaDB collection.
3. Retrieve semantically related memories before each response.
4. Add that context to the prompt sent to the NVIDIA-hosted language model.
5. Generate and print the assistant's reply.

The local `chroma_db/` directory makes the memory persistent between runs.

## Retention Policy

- **Recency half-life:** 24 hours. A memory's recency score decays to 0.5
  after 24 hours.
- **Relevance and recency weighting:** the combined score uses 0.7 relevance
  and 0.3 recency. Relevance gets the larger weight because matching the
  current question is the primary signal; recency still helps prefer fresher
  memories when several are similarly relevant.
- **Forget versus fading:** `/forget <topic>` is an explicit deletion request.
  It searches for related memories and deletes matches. Ordinary memories are
  not deleted just because they become old; they gradually receive a lower
  recency score instead.
- **Count-based compaction:** once the memory count is above 3, the lowest
  scoring memories are summarized and replaced with one compact system memory.
  A count threshold keeps the store bounded and prevents compaction from
  running on every message.

Assistant replies are deliberately never stored as memories. This was a real
bug caught and fixed tonight: storing generated replies could feed a
hallucination back into later prompts and compound it over time. Normal chat
therefore stores user messages only; compaction summaries are the exception
because they are explicitly generated as memory summaries.

## Known Limitations

Forget-command matching uses embedding similarity plus a distance threshold; on a small memory corpus this occasionally under- or over-matches, since semantically related statements (e.g. 'I like hiking' and '/forget hiking') can sit close together in embedding space. A production version would likely combine this with exact entity extraction rather than pure similarity.

## API Failure Handling

The NVIDIA API client uses a 20-second request timeout and retries failed
requests twice, for up to three attempts total. If all attempts fail, it
returns a temporary-unavailable message so the CLI can degrade gracefully
instead of crashing. The chat loop also catches unexpected model errors and
continues waiting for the next user message.

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

Type `exit` to end the conversation.

## Project Structure

| File              | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `chat.py`         | Command-line chat loop and context retrieval |
| `memory_store.py` | Embeddings and ChromaDB persistence          |
| `llm_client.py`   | NVIDIA API client and retry handling         |
| `retention.py`    | Recency scoring and compaction helpers       |

## License

This project does not currently include a license.
