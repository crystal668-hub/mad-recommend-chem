"""
RAG system (retrieval adapter).

This project stores literature chunks in Chroma (via `database/vector_store.py`).
Historically we also had a LlamaIndex-based RAG implementation here, but that created
two parallel indexing/query stacks (and required LlamaIndex "docstore.json" state
that is NOT produced by `build_vector_db.py`).

Current design:
- Keep Chroma/VectorStore as the single source of truth.
- `RAGSystem` is a thin adapter that:
  1) embeds the query (using the project embedder / provider config),
  2) runs vector similarity search in Chroma,
  3) returns results in the shape expected by agents: [{text, score, metadata}, ...].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from database.vector_store import VectorStore
from utils.logger import Logger

logger = Logger.create_module_logger("database.rag_system")


class RAGSystem:
    """
    Chroma-backed retrieval adapter.

    Required capability for agents:
    - `retrieve(query: str) -> List[{text, score, metadata}]`

    Notes:
    - Collections in this repo are built by `build_vector_db.py` using precomputed
      embeddings. Therefore Chroma cannot embed `query_texts` automatically; we MUST
      supply `query_embedding` for retrieval.
    - The embedder is injected so we can match the embedding model used to build the
      collection (agent1/agent2/agent3/agent4 can each have different embeddings).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder: Any,
        agent_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        distance_metric: str = "cosine",
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.collection_name = str(collection_name)
        self.agent_name = (agent_name or "").strip() or None
        self.top_k = int(top_k)
        self.similarity_threshold = float(similarity_threshold) if similarity_threshold is not None else None
        self.distance_metric = (distance_metric or "cosine").strip().lower()

        self.embedder = embedder
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=None,  # we provide query embeddings explicitly
            distance_metric=self.distance_metric,
        )

    # -------------------------
    # Public API
    # -------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Chroma.

        Returns:
            List[Dict]: each item has:
              - text: chunk text
              - score: similarity score when computable
              - metadata: stored metadata (must include doc_id + chunk_id for `source_id`)
        """
        q = (query or "").strip()
        if not q:
            return []

        k = int(top_k) if top_k is not None else self.top_k
        k = max(1, k)

        query_embedding = self._embed_query(q)
        similar_docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            threshold=None,  # apply similarity_threshold ourselves on normalized "score"
            where=where,
        )

        results: List[Dict[str, Any]] = []
        for item in similar_docs or []:
            distance = item.get("distance")
            score = None
            if distance is not None:
                try:
                    # For cosine distance, a common mapping is similarity = 1 - distance.
                    score = 1.0 - float(distance)
                except Exception:
                    score = None

            if self.similarity_threshold is not None and score is not None:
                if score < self.similarity_threshold:
                    continue

            results.append(
                {
                    "text": item.get("document") or "",
                    "score": score,
                    "metadata": item.get("metadata") or {},
                }
            )

        return results

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Compatibility shim.

        The previous implementation returned an LLM-generated answer. In the current
        stack, the LLM lives in the agent runtime; this RAG component is retrieval-only.
        """
        nodes = self.retrieve(query_text)
        return {"answer": "", "source_nodes": nodes}

    def get_index_stats(self) -> Dict[str, Any]:
        try:
            doc_count = self.vector_store.get_collection_count()
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {
            "status": "initialized",
            "collection_name": self.collection_name,
            "document_count": doc_count,
            "top_k": self.top_k,
        }

    # -------------------------
    # Internals
    # -------------------------

    def _embed_query(self, text: str) -> List[float]:
        """
        Embed a query string using the injected embedder.

        Supported embedder shapes:
        - embedder.embed_query(text, agent_name=...)  (preferred)
        - embedder.embed_text(text, agent_name=...)
        - embedder(text) -> List[float]
        """
        if self.embedder is None:
            raise RuntimeError("RAGSystem embedder is not configured")

        # Prefer explicit query embedding if available (e.g. Voyage supports input_type='query').
        if hasattr(self.embedder, "embed_query"):
            try:
                return self.embedder.embed_query(text, agent_name=self.agent_name)  # type: ignore[attr-defined]
            except TypeError:
                # Some implementations may not accept agent_name.
                return self.embedder.embed_query(text)  # type: ignore[attr-defined]

        if hasattr(self.embedder, "embed_text"):
            try:
                return self.embedder.embed_text(text, agent_name=self.agent_name)  # type: ignore[attr-defined]
            except TypeError:
                return self.embedder.embed_text(text)  # type: ignore[attr-defined]

        if callable(self.embedder):
            return self.embedder(text)

        raise TypeError("Unsupported embedder type; expected embed_query/embed_text/callable")
