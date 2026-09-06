import chromadb
from sentence_transformers import SentenceTransformer
import time
import uuid

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection("memories")

def embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()

def add_memory(text: str, role: str) -> str:
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