import chromadb
import time
import uuid
import os
from dotenv import load_dotenv
from openai import OpenAI

_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection("memories")

load_dotenv()

_embed_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

def embed(text: str) -> list[float]:
    response = _embed_client.embeddings.create(
        input=[text],
        model="nvidia/nemotron-3-embed-1b",
    )
    return response.data[0].embedding


def add_memory(text: str, role: str) -> str:
    existing = get_all_memories()["documents"]
    if text in existing:
        return None
    mem_id = str(uuid.uuid4())
    _collection.add(
        ids=[mem_id],
        embeddings=[embed(text)],
        documents=[text],
        metadatas=[{"role": role, "timestamp": time.time(), "access_count": 0}],
    )
    return mem_id

def search_memories(query: str, k: int = 5):
    results = _collection.query(query_embeddings=[embed(query)], n_results=k)
    # returns lists of ids, documents, metadatas, distances — zip them together
    return list(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))

def get_all_memories():
    return _collection.get(include=["documents", "metadatas", "embeddings"])

def delete_memories(ids: list[str]):
    if ids:
        _collection.delete(ids=ids)

def update_memory(mem_id: str, new_text: str, metadata: dict):
    _collection.update(
        ids=[mem_id],
        embeddings=[embed(new_text)],
        documents=[new_text],
        metadatas=[metadata],
    )

def is_question(text: str) -> bool:
    text = text.strip().lower()
    if text.endswith('?'):
        return True
    starters = ('what', 'who', 'when', 'where', 'why', 'how', 'do ', 'does ', 'is ', 'are ', 'can ', 'could ', 'would ', 'should ')
    return text.startswith(starters)