# Agentic Memory Assistant

A CLI-based conversational agent with retrieval-augmented memory: it
embeds and stores what you tell it, retrieves relevant past statements
on each new turn, and supports explicit forgetting and compaction —
built to understand how memory-augmented agents actually work, not to
demo a finished product.

## Architecture

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) — small
  and fast, chosen for short conversational memories over accuracy on
  longer documents.
- **Vector store**: Chroma, local and persistent — chosen over
  pgvector for zero server setup under a tight timeline.
- **LLM**: NVIDIA NIM API (`mistralai/mistral-nemotron`), OpenAI-
  compatible endpoint. Ollama (local, `llama3.2:3b`) is supported as a
  dev-time swap for faster iteration — see `USE_LOCAL` in
  `llm_client.py` — but production always points at NIM.
- **Interface**: CLI (`chat.py`).

## Retention policy — design decisions

- **Recency**: exponential decay, 24-hour half-life. A memory is worth
  half as much every 24 hours it goes unused.
- **Relevance**: `1 / (1 + distance)`, converting Chroma's distance
  metric into a 0–1 relevance score.
- **Combined score**: `0.7 * relevance + 0.3 * recency`. Relevance is
  weighted higher because the point of retrieval is surfacing what's
  actually useful to the current query; recency breaks ties rather
  than dominating.
- **Compaction**: count-based — above 50 stored memories, the lowest-
  scoring 20% are summarized via one LLM call and replaced with the
  summary. Compaction ranks by recency only (no active query exists at
  that point, so relevance isn't computable there).
- **Forgetting — two-tier by design**: `/forget X` deletes only memories
  containing the literal term or within a tight similarity threshold
  (distance < 0.35) — precise, so forgetting one topic doesn't sweep up
  loosely related ones. `forget everything about/related to X` uses a
  looser threshold (distance < 0.9) to also catch near-paraphrases.
  Verified against a three-fact test case: an unrelated fact (distance
  1.83) and a near-paraphrase (distance 0.83) — narrow mode only
  deletes the exact match, broad mode also removes the paraphrase, the
  unrelated fact survives both.

## Bugs found and fixed during development

- **Hallucination compounding**: originally, both user input and
  assistant replies were stored as memories. The LLM sometimes
  fabricates plausible-sounding details to keep a conversation going
  (e.g. inventing a "trip to the Rocky Mountains" never mentioned) —
  storing those replies let fabricated facts get retrieved and treated
  as ground truth on later turns, compounding over time. Fixed by only
  ever persisting user-stated input.
- **Questions stored as facts**: user questions (e.g. "what outdoor
  activities do I like?") were being embedded and stored identically
  to statements, polluting future retrieval. Fixed with a heuristic
  question filter (trailing `?` or common question-word start) that
  skips storage for questions.

## Known limitations

- Question detection is heuristic, not semantic — a message like "I
  like ramen, what do you think?" ends in `?` and would be skipped
  entirely, losing the real fact it contains.
- No duplicate-entry detection yet — repeating the same statement
  twice creates two separate memories.
- Forget's tight/loose thresholds (0.35 / 0.9) were tuned against one
  test case, not a broad benchmark — they may need adjustment for
  other topics or corpus sizes.
- NIM's free tier was intermittently slow/unreliable during
  development (multiple timeouts and 500s observed); a retry-with-
  backoff wrapper is in place, but sustained outages will still
  degrade the experience.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "NVIDIA_API_KEY=your_key_here" > .env
python chat.py
```
