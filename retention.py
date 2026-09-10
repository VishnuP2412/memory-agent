import time
import re

RECENCY_HALF_LIFE_SECONDS = 60 * 60 * 24    #24Hours
RELEVANCE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3
COMPACTION_THRESHOLD = 3   #The compaction will/should be triggered if there are above these many messages
COMPACTION_FRACTION = 0.2

def recency_score(timestamp: float) -> float:
    age = time.time() - timestamp
    return 0.5 ** (age / RECENCY_HALF_LIFE_SECONDS)

def relevance_score(distance: float) -> float:
    return 1 / (1 + distance)

def combined_score(timestamp: float, distance: float) -> float:
     return (RELEVANCE_WEIGHT * relevance_score(distance)
            + RECENCY_WEIGHT * recency_score(timestamp))

def is_forget_command(user_input: str) -> str | None:
    text = user_input.strip()
    broad = re.match(r'^/?forget (?:everything|anything)\s+(?:about|related to)\s+(.+)$', text, re.IGNORECASE)
    if broad:
        return broad.group(1), True
    narrow = re.match(r'^/forget\s+(.+)$', text, re.IGNORECASE)
    if narrow:
        return narrow.group(1), False
    fallback = re.match(r"forget (?:that )?(?:i said |about )?(.+)", text, re.IGNORECASE)
    if fallback:
        return fallback.group(1), False
    return None, False

def should_compact(all_memories) -> bool:
    return len(all_memories) > COMPACTION_THRESHOLD

def compact_memories(low_score_memories: list[str], ask_llm_fn) -> str:
    joined = "\n".join(f"- {m}" for m in low_score_memories)
    prompt = (
        "Summarize these old, low-relevance conversation memories into one "
        "short paragraph, preserving only facts likely to matter later:\n\n"
        f"{joined}"
    )
    return ask_llm_fn([{"role": "user", "content": prompt}])