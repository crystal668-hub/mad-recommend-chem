import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from database.embedding_runtime import (
    EmbeddingFailure,
    EmbeddingWriteItem,
    EmbeddingWritePipeline,
    write_failure_manifest,
)


class FakeVectorStore:
    def __init__(self, fail=False, delay=0.0):
        self.fail = fail
        self.delay = delay
        self.calls = []
        self.thread_ids = []

    def add_documents(self, documents, embeddings, metadatas, ids):
        if self.delay:
            time.sleep(self.delay)
        self.thread_ids.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("database is unavailable")
        self.calls.append(
            {
                "documents": list(documents),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
                "ids": list(ids),
            }
        )


def _item(store, agent, start, count):
    ids = [f"{agent}-{index}" for index in range(start, start + count)]
    return EmbeddingWriteItem(
        agent_name=agent,
        collection_name=f"collection_{agent}",
        provider="test-provider",
        model="test-model",
        quota_group="test-quota",
        vector_store=store,
        texts=[f"text-{chunk_id}" for chunk_id in ids],
        embeddings=[[float(index + 1), 1.0] for index in range(count)],
        metadatas=[{"chunk_id": chunk_id} for chunk_id in ids],
        ids=ids,
    )


class EmbeddingWritePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_coalesces_batches_and_uses_one_writer_thread(self):
        store = FakeVectorStore()
        pipeline = EmbeddingWritePipeline(
            write_batch_size=3,
            queue_max_batches=2,
            flush_interval_ms=1000,
        )
        await pipeline.start()
        await pipeline.submit(_item(store, "agent1", 0, 2))
        await pipeline.submit(_item(store, "agent1", 2, 2))
        await pipeline.close()

        self.assertEqual([len(call["ids"]) for call in store.calls], [3, 1])
        self.assertEqual(pipeline.newly_added["agent1"], 4)
        self.assertEqual(len(set(store.thread_ids)), 1)
        self.assertEqual(pipeline.failures, [])

    async def test_different_collections_are_never_mixed(self):
        first = FakeVectorStore()
        second = FakeVectorStore()
        pipeline = EmbeddingWritePipeline(write_batch_size=10, flush_interval_ms=1000)
        await pipeline.start()
        await pipeline.submit(_item(first, "agent1", 0, 2))
        await pipeline.submit(_item(second, "agent2", 0, 2))
        await pipeline.close()

        self.assertEqual(first.calls[0]["ids"], ["agent1-0", "agent1-1"])
        self.assertEqual(second.calls[0]["ids"], ["agent2-0", "agent2-1"])

    async def test_flush_timeout_writes_partial_batch_before_close(self):
        store = FakeVectorStore()
        pipeline = EmbeddingWritePipeline(write_batch_size=100, flush_interval_ms=20)
        await pipeline.start()
        await pipeline.submit(_item(store, "agent1", 0, 1))
        await asyncio.sleep(0.08)

        self.assertEqual(len(store.calls), 1)
        self.assertEqual(pipeline.newly_added["agent1"], 1)
        await pipeline.close()

    async def test_write_failure_does_not_increment_success_count(self):
        store = FakeVectorStore(fail=True)
        pipeline = EmbeddingWritePipeline(write_batch_size=2, flush_interval_ms=1000)
        await pipeline.start()
        await pipeline.submit(_item(store, "agent1", 0, 2))
        await pipeline.close()

        self.assertEqual(pipeline.newly_added.get("agent1", 0), 0)
        self.assertEqual(len(pipeline.failures), 2)
        self.assertTrue(all(failure.error_type == "vector_store_write" for failure in pipeline.failures))
        self.assertTrue(all(failure.chunk_id.startswith("agent1-") for failure in pipeline.failures))

    async def test_queue_is_bounded(self):
        pipeline = EmbeddingWritePipeline(write_batch_size=10, queue_max_batches=1)
        self.assertEqual(pipeline.queue.maxsize, 1)
        await pipeline.start()
        await pipeline.close()


class FailureManifestTests(unittest.TestCase):
    def test_manifest_is_jsonl_and_sanitizes_secrets(self):
        failure = EmbeddingFailure(
            agent="agent1",
            collection="collection_agent1",
            chunk_id="chunk-1",
            provider="zenmux",
            model="model",
            quota_group="shared",
            attempts=2,
            error_type="rate_limit",
            retryable=True,
            message="Authorization: Bearer secret-value",
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "failures.jsonl"
            result = write_failure_manifest(target, [failure])
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(result, target)
        self.assertEqual(payload["chunk_id"], "chunk-1")
        self.assertNotIn("secret-value", payload["message"])


if __name__ == "__main__":
    unittest.main()
