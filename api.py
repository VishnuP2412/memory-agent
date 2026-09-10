# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from memory_store import add_memory, search_memories, get_all_memories, delete_memories
from retention import is_forget_command, is_question
from llm_client import ask_llm

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(msg: Message):
    user_input = msg.text.strip()

    forget_topic, is_broad = is_forget_command(user_input)
    if forget_topic:
        matches = search_memories(forget_topic, k=20)
        threshold = 0.9 if is_broad else 0.35
        ids_to_delete = [
            mid for mid, doc, meta, dist in matches
            if forget_topic.lower() in doc.lower() or dist < threshold
        ]
        delete_memories(ids_to_delete)
        return {"reply": f"Forgot memories related to '{forget_topic}'."}

    results = search_memories(user_input, k=5)
    context = "Relevant past context:\n" + "\n".join(f"- {doc}" for _id, doc, meta, dist in results) if results else ""
    system_prompt = "You are a helpful assistant with access to memory of past conversation."
    if context:
        system_prompt += "\n\n" + context

    reply = ask_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ])

    if not is_question(user_input):
        add_memory(user_input, "user")

    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}