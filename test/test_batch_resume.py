import uuid
import unittest
import sys
from pathlib import Path


try:
    import chromadb  # noqa: F401
except Exception:
    chromadb = None  # type: ignore


VectorStore = None
if chromadb is not None:
    # Ensure project root is on sys.path when running via `python -m unittest discover`.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    # Only import VectorStore when chromadb is available (VectorStore depends on it).
    from database.vector_store import VectorStore  # type: ignore


@unittest.skipIf(chromadb is None, "chromadb is not installed; resume integration test skipped")
class BatchResumeTests(unittest.TestCase):
    def test_resume_adds_only_missing_chunks(self):
        # Use a repo-local directory to avoid OS temp path issues on some setups.
        persist_dir = Path(".cache") / f"chroma_resume_test_{uuid.uuid4().hex[:8]}"
        persist_dir.mkdir(parents=True, exist_ok=True)

        store = VectorStore(
            persist_directory=str(persist_dir),
            collection_name="resume_test",
            embedding_function=None,
            distance_metric="cosine",
        )
        store.reset_collection()

        total = 20
        texts = [f"text {i}" for i in range(total)]
        embeddings = [[float(i), float(i) + 0.1, 0.0, 0.0] for i in range(total)]
        metadatas = [{"doc_id": "10.0000/test", "chunk_id": i, "reaction_type": "HER"} for i in range(total)]
        ids = [f"10.0000/test#chunk:{i}" for i in range(total)]

        # Simulate an interrupted run that wrote only the first 7 chunks.
        store.add_documents(
            documents=texts[:7],
            embeddings=embeddings[:7],
            metadatas=metadatas[:7],
            ids=ids[:7],
        )
        self.assertEqual(store.get_collection_count(), 7)

        # Resume: only add missing ids, in batches (mirrors build_vector_db_batch.py behavior).
        newly_added = 0
        batch_size = 5
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            b_ids = ids[start:end]
            b_texts = texts[start:end]
            b_embs = embeddings[start:end]
            b_metas = metadatas[start:end]

            existing = store.get_existing_ids(b_ids)
            missing_idx = [i for i, cid in enumerate(b_ids) if cid not in existing]
            if not missing_idx:
                continue

            add_docs = [b_texts[i] for i in missing_idx]
            add_embs = [b_embs[i] for i in missing_idx]
            add_metas = [b_metas[i] for i in missing_idx]
            add_ids = [b_ids[i] for i in missing_idx]

            store.add_documents(
                documents=add_docs,
                embeddings=add_embs,
                metadatas=add_metas,
                ids=add_ids,
            )
            newly_added += len(add_ids)

        self.assertEqual(newly_added, 13)
        self.assertEqual(store.get_collection_count(), 20)


if __name__ == "__main__":
    unittest.main()
