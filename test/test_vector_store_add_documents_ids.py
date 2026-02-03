import hashlib
import sys
import types
import unittest
from pathlib import Path


def _install_chromadb_stub() -> None:
    """Install a minimal chromadb stub so importing database.vector_store works in minimal envs."""
    chromadb = types.ModuleType("chromadb")
    chromadb_config = types.ModuleType("chromadb.config")

    class Settings:  # noqa: D401 - tiny stub
        def __init__(self, **_kwargs):
            pass

    class PersistentClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get_or_create_collection(self, *args, **kwargs):
            raise RuntimeError("chromadb stub: collection access not supported in this test")

        def delete_collection(self, *args, **kwargs):
            return None

    chromadb.PersistentClient = PersistentClient
    chromadb_config.Settings = Settings

    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.config"] = chromadb_config


# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Optional dependency (tests don\'t need real Chroma).
try:
    import chromadb  # type: ignore  # noqa: F401
except Exception:
    _install_chromadb_stub()

from database.vector_store import VectorStore


class DummyCollection:
    def __init__(self):
        self.calls = []

    def add(self, documents, embeddings, metadatas, ids):
        self.calls.append(
            {
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "ids": ids,
            }
        )


def _new_store() -> VectorStore:
    # Bypass __init__ (which requires chromadb); we only test add_documents id/metadata logic.
    vs = VectorStore.__new__(VectorStore)
    vs.collection_name = "unit_test"
    vs.collection = DummyCollection()
    return vs


class VectorStoreAddDocumentsIdsTests(unittest.TestCase):
    def test_doi_chunks_use_doi_chunk_ids(self):
        vs = _new_store()

        docs = ["chunk0", "chunk1"]
        metas = [
            {"doc_id": "10.1000/abc.def", "chunk_id": 0, "reaction_type": "HER"},
            {"doc_id": "10.1000/abc.def", "chunk_id": "1", "reaction_type": "HER"},
        ]
        embs = [[0.0, 0.1], [0.1, 0.0]]

        vs.add_documents(documents=docs, metadatas=metas, embeddings=embs)

        self.assertEqual(len(vs.collection.calls), 1)
        call = vs.collection.calls[0]
        ids = call["ids"]
        self.assertEqual(ids, ["10.1000/abc.def#chunk:0", "10.1000/abc.def#chunk:1"])
        self.assertEqual(len(ids), len(set(ids)))

        out_meta0 = call["metadatas"][0]
        out_meta1 = call["metadatas"][1]
        self.assertEqual(out_meta0["chunk_index"], 0)
        self.assertEqual(out_meta1["chunk_index"], 1)
        self.assertEqual(out_meta0["chunk_id"], "10.1000/abc.def#chunk:0")
        self.assertEqual(out_meta1["chunk_id"], "10.1000/abc.def#chunk:1")

    def test_no_doi_chunks_use_content_hash_ids(self):
        vs = _new_store()

        docs = ["hello"]
        metas = [{"doc_id": "no-doi:her/x", "chunk_id": 0, "reaction_type": "HER"}]
        embs = [[0.0, 0.1]]

        vs.add_documents(documents=docs, metadatas=metas, embeddings=embs)

        call = vs.collection.calls[0]
        ids = call["ids"]
        expected = "hash_" + hashlib.sha256(b"hello").hexdigest()
        self.assertEqual(ids, [expected])
        self.assertEqual(call["metadatas"][0]["chunk_index"], 0)
        self.assertEqual(call["metadatas"][0]["chunk_id"], expected)

    def test_duplicate_doi_chunk_ids_are_disambiguated(self):
        vs = _new_store()

        docs = ["chunk A", "chunk B"]
        metas = [
            {"doc_id": "10.1000/dup", "chunk_id": 0, "reaction_type": "OER"},
            {"doc_id": "10.1000/dup", "chunk_id": 0, "reaction_type": "OER"},
        ]
        embs = [[0.0, 0.1], [0.1, 0.0]]

        vs.add_documents(documents=docs, metadatas=metas, embeddings=embs)

        call = vs.collection.calls[0]
        ids = call["ids"]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "10.1000/dup#chunk:0")
        self.assertTrue(ids[1].startswith("10.1000/dup#chunk:0#dup:"))

        # Metadata should mirror ids.
        self.assertEqual(call["metadatas"][0]["chunk_id"], ids[0])
        self.assertEqual(call["metadatas"][1]["chunk_id"], ids[1])
        self.assertEqual(call["metadatas"][0]["chunk_index"], 0)
        self.assertEqual(call["metadatas"][1]["chunk_index"], 0)

    def test_duplicate_hash_chunk_ids_are_disambiguated(self):
        vs = _new_store()

        docs = ["same", "same"]
        metas = [
            {"doc_id": "no-doi:a", "chunk_id": 0, "reaction_type": "HER"},
            {"doc_id": "no-doi:b", "chunk_id": 1, "reaction_type": "HER"},
        ]
        embs = [[0.0, 0.1], [0.1, 0.0]]

        vs.add_documents(documents=docs, metadatas=metas, embeddings=embs)

        call = vs.collection.calls[0]
        ids = call["ids"]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids[1].startswith(ids[0] + "#dup:"))

        self.assertEqual(call["metadatas"][0]["chunk_index"], 0)
        self.assertEqual(call["metadatas"][1]["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
