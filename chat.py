from retention import is_forget_command, should_compact, combined_score, compact_memories, recency_score
from memory_store import add_memory, search_memories, get_all_memories, delete_memories, is_question
from llm_client import ask_llm

def build_context(query: str) -> str:
    results = search_memories(query, k=5)
    if not results:
        return ""
    lines = [doc for (_id, doc, meta, dist) in results]
    print(f"DEBUG: context sent to model = {lines!r}")
    return "Relevant past context:\n" + "\n".join(f"- {l}" for l in lines)

def main():
    print("Memory agent CLI. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        forget_topic, is_broad = is_forget_command(user_input)
        if forget_topic:
            matches = search_memories(forget_topic, k=20)
            threshold = 0.9 if is_broad else 0.35
            ids_to_delete = [
                mid for mid, doc, meta, dist in matches
                if forget_topic.lower() in doc.lower() or dist < threshold
            ]
            delete_memories(ids_to_delete)
            print(f"Agent: Forgot memories related to '{forget_topic}'.\n")
            continue

        context = build_context(user_input)

        system_prompt = "You are a helpful assistant with access to memory of past conversation."
        if context:
            system_prompt += "\n\n" + context

        try:
            reply = ask_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ])
            print(f"Agent: {reply}\n")
        except Exception as e:
            print(f"Agent: (had trouble reaching the model, try again — {e})\n")
            continue

        if not is_question(user_input):
            add_memory(user_input, "user")

        all_mem = get_all_memories()
        if should_compact(all_mem["ids"]):
            scored = [
                (mid, combined_score(meta["timestamp"],0.0))
                for mid, meta in zip(all_mem["ids"], all_mem["metadatas"])
            ]
            scored.sort(key=lambda x: x[1])
            low_score_ids = [mid for mid, _ in scored[:10]]  # lowest 10, or use your COMPACTION_FRACTION
            low_score_texts = [doc for doc, mid in zip(all_mem["documents"], all_mem["ids"]) if mid in low_score_ids]

            summary = compact_memories(low_score_texts, ask_llm)
            delete_memories(low_score_ids)
            add_memory(summary, "system")

if __name__ == "__main__":
    main()