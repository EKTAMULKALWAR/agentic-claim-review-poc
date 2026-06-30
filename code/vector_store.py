"""
Vector store for payer-policy semantic search.
Uses ChromaDB (persistent, embedded) + sentence-transformers (all-MiniLM-L6-v2).

Production equivalent: Pinecone / pgvector + OpenAI/Cohere embeddings.
"""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from policies import POLICIES

_PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
_COLLECTION  = "payer_policies"
_EMBED_MODEL = "all-MiniLM-L6-v2"


class PolicyVectorStore:
    """Persistent ChromaDB collection of payer policies with semantic search."""

    def __init__(self) -> None:
        ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=_PERSIST_DIR)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._sync()

    # ── internal ──────────────────────────────────────────────────────────────

    def _sync(self) -> None:
        """Upsert all policies so the index stays current with policies.py."""
        self._col.upsert(
            ids       = [p["id"] for p in POLICIES],
            documents = [p["text"] for p in POLICIES],
            metadatas = [
                {"id": p["id"], "title": p["title"],
                 "source": p.get("source", ""), "effective": p.get("effective", "")}
                for p in POLICIES
            ],
        )

    # ── public ────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 1) -> list[dict]:
        """
        Semantic search over the policy collection.
        Returns a list of policy dicts with an added `similarity_score` key.
        """
        n = min(n_results, len(POLICIES))
        results = self._col.query(query_texts=[query], n_results=n)

        out = []
        for i, pid in enumerate(results["ids"][0]):
            policy = next(p for p in POLICIES if p["id"] == pid)
            distance   = results["distances"][0][i]
            similarity = round(1.0 - distance, 4)        # cosine distance → similarity
            out.append({**policy, "similarity_score": similarity})
        return out

    @property
    def model_name(self) -> str:
        return _EMBED_MODEL

    @property
    def policy_count(self) -> int:
        return self._col.count()
