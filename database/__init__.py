"""
database package.

Keep imports lightweight: some modules (e.g. VectorStore) depend on optional heavy
dependencies (chromadb). Tests/scripts that only need `database.text_processor`
should not be forced to import those dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["VectorStore", "RAGSystem"]

if TYPE_CHECKING:
    from database.rag_system import RAGSystem
    from database.vector_store import VectorStore


def __getattr__(name: str) -> Any:
    # Lazy export to avoid importing chromadb unless it's actually needed.
    if name == "VectorStore":
        from database.vector_store import VectorStore

        return VectorStore
    if name == "RAGSystem":
        from database.rag_system import RAGSystem

        return RAGSystem
    raise AttributeError(f"module 'database' has no attribute {name!r}")

