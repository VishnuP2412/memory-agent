from memory_store import add_memory, search_memories
from llm_client import ask_llm

def build_context(query: str) -> str:
    results = search_memories(query, k=5)
    if not results:
        return ""
    lines = [doc for (_id, doc, meta, dist) in results]
    return "Relevant past context:\n" + "\n".join(f"- {l}" for l in lines)

def main():
    print("Memory agent CLI. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        context = build_context(user_input)
        system_prompt = "You are a helpful assistant with access to memory of past conversation."
        if context:
            system_prompt += "\n\n" + context

        reply = ask_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ])
        print(f"Agent: {reply}\n")

        add_memory(user_input, "user")
        add_memory(reply, "assistant")

if __name__ == "__main__":
    main()