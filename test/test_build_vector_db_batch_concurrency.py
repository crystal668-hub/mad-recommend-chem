import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import build_vector_db_batch as batch
from database.embedding_runtime import (
    AgentEmbeddingRuntime,
    EmbeddingRuntimeSettings,
    QuotaPolicy,
    RetryPolicy,
    build_text_batches,
)


class FakeVectorStore:
    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.existing_queries = []
        self.add_calls = []

    def get_existing_ids(self, ids):
        self.existing_queries.append(list(ids))
        return set(ids).intersection(self.existing)

    def add_documents(self, documents, embeddings, metadatas, ids):
        self.add_calls.append(
            {
                "documents": list(documents),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
                "ids": list(ids),
            }
        )

    def get_collection_count(self):
        return sum(len(call["ids"]) for call in self.add_calls) + len(self.existing)


class FakeEmbedder:
    def __init__(self, dimension=2, fail_text=None, tracker=None):
        self.dimension = dimension
        self.fail_text = fail_text
        self.calls = []
        self.tracker = tracker

    async def embed_documents_batch(self, texts, agent_name):
        self.calls.append((agent_name, list(texts)))
        if self.tracker is not None:
            self.tracker["inflight"] += 1
            self.tracker["peak"] = max(self.tracker["peak"], self.tracker["inflight"])
            await asyncio.sleep(0.02)
            self.tracker["inflight"] -= 1
        if self.fail_text and self.fail_text in texts:
            raise RuntimeError("provider failed")
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


class FailFastEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.slow_started = asyncio.Event()
        self.slow_cancelled = asyncio.Event()
        self.closed = False

    async def embed_documents_batch(self, texts, agent_name):
        self.calls.append((agent_name, list(texts)))
        if "fail" in texts:
            await self.slow_started.wait()
            raise RuntimeError("provider failed")
        self.slow_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.slow_cancelled.set()
            raise
        return [[1.0, 1.0] for _ in texts]

    async def aclose(self):
        self.closed = True


def _settings(batch_size=2, concurrency=1):
    return EmbeddingRuntimeSettings(
        global_max_inflight=concurrency,
        write_batch_size=100,
        write_queue_max_batches=2,
        write_flush_interval_ms=10,
        retry_policy=RetryPolicy(max_attempts=1),
        quota_groups={
            "shared": QuotaPolicy(
                initial_inflight=concurrency,
                max_inflight=concurrency,
                cooldown_seconds=0,
            )
        },
        agents={
            "agent1": AgentEmbeddingRuntime("shared", batch_size, batch_size, None),
            "agent3": AgentEmbeddingRuntime("shared", batch_size, batch_size, None),
        },
    )


def _context(agent, embedder, store, texts, batch_size=2):
    ids = [f"{agent}-{index}" for index in range(len(texts))]
    batches = build_text_batches(
        agent_name=agent,
        texts=texts,
        ids=ids,
        metadatas=[{"chunk_id": chunk_id} for chunk_id in ids],
        max_items=batch_size,
    )
    return batch._AgentEmbeddingContext(
        agent_name=agent,
        collection_name=f"collection_{agent}",
        provider="zenmux",
        model="test-model",
        dimension=2,
        quota_group="shared",
        embedder=embedder,
        vector_store=store,
        batches=batches,
        already_present=0,
    )


class MissingIdTests(unittest.TestCase):
    def test_existing_ids_are_checked_in_large_batches(self):
        ids = [f"id-{index}" for index in range(2501)]
        store = FakeVectorStore(existing={"id-5", "id-2000"})

        missing, present = batch._get_missing_indices(store, ids, batch_size=1000)

        self.assertEqual([len(query) for query in store.existing_queries], [1000, 1000, 501])
        self.assertEqual(present, 2)
        self.assertNotIn(5, missing)
        self.assertNotIn(2000, missing)
        self.assertEqual(len(missing), 2499)


class EmbeddingPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_one_still_uses_request_batches(self):
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        context = _context("agent1", embedder, store, ["a", "b", "c", "d"], batch_size=2)

        outcome = await batch._run_embedding_pipeline([context], _settings(2, 1))

        self.assertEqual(embedder.calls, [("agent1", ["a", "b"]), ("agent1", ["c", "d"])])
        self.assertEqual(outcome.agent_results["agent1"]["newly_added"], 4)
        self.assertEqual(outcome.agent_results["agent1"]["status"], "ok")

    async def test_shared_quota_caps_agents_together(self):
        tracker = {"inflight": 0, "peak": 0}
        settings = _settings(batch_size=1, concurrency=2)
        first = _context("agent1", FakeEmbedder(tracker=tracker), FakeVectorStore(), ["a", "b", "c"], 1)
        second = _context("agent3", FakeEmbedder(tracker=tracker), FakeVectorStore(), ["d", "e", "f"], 1)

        outcome = await batch._run_embedding_pipeline([first, second], settings)

        self.assertEqual(tracker["peak"], 2)
        self.assertEqual(outcome.scheduler_snapshot["shared"]["peak_inflight"], 2)

    async def test_failed_batch_is_not_written_and_result_is_partial(self):
        embedder = FakeEmbedder(fail_text="bad")
        store = FakeVectorStore()
        context = _context("agent1", embedder, store, ["good", "bad"], batch_size=1)

        outcome = await batch._run_embedding_pipeline([context], _settings(1, 1))

        written_ids = [chunk_id for call in store.add_calls for chunk_id in call["ids"]]
        self.assertEqual(written_ids, ["agent1-0"])
        self.assertEqual(outcome.agent_results["agent1"]["status"], "partial")
        self.assertEqual(outcome.agent_results["agent1"]["failed"], 1)
        self.assertEqual([failure.chunk_id for failure in outcome.failures], ["agent1-1"])

    async def test_fail_fast_cancels_outstanding_batches_before_closing_clients(self):
        embedder = FailFastEmbedder()
        store = FakeVectorStore()
        context = _context("agent1", embedder, store, ["slow", "fail"], batch_size=1)

        outcome = await batch._run_embedding_pipeline(
            [context],
            _settings(batch_size=1, concurrency=2),
            continue_on_error=False,
        )

        self.assertTrue(embedder.slow_cancelled.is_set())
        self.assertTrue(embedder.closed)
        self.assertEqual(store.add_calls, [])
        self.assertEqual(outcome.agent_results["agent1"]["status"], "error")
        self.assertEqual([failure.chunk_id for failure in outcome.failures], ["agent1-1"])


class BatchBuilderEntryPointTests(unittest.TestCase):
    def test_full_builder_uses_runtime_batching_and_writer(self):
        class FakeConfig:
            def __init__(self):
                self.config = {
                    "llm": {
                        "agent1": {
                            "model": "chat-model",
                            "embedding_model": "test-embedding",
                            "embedding_provider": "zenmux",
                            "embedding_transport": "openai_compatible",
                            "embedding_quota_group": "myrimate",
                            "embedding_request_batch_size": 2,
                            "embedding_max_batch_items": 2,
                            "emb_url": "https://gateway.example/v1",
                            "api_key": "test-key",
                        }
                    },
                    "embedding_runtime": {
                        "global_max_inflight": 1,
                        "write_batch_size": 3,
                        "write_flush_interval_ms": 5,
                        "retry": {"max_attempts": 1},
                        "quota_groups": {
                            "myrimate": {"initial_inflight": 1, "max_inflight": 1}
                        },
                    },
                    "vector_store": {
                        "persist_directory": "unused",
                        "collection_name": "test_collection",
                        "distance_metric": "cosine",
                    },
                    "rag": {"chunk_size": 10, "chunk_overlap": 0},
                }

            def get_vector_store_config(self):
                return self.config["vector_store"]

            def get_rag_config(self):
                return self.config["rag"]

            def get_llm_config(self, name):
                return dict(self.config["llm"][name])

        class FakeProcessor:
            def __init__(self, data_dir):
                self.data_dir = data_dir

            def load_literature_type_documents(self, **kwargs):
                return [SimpleNamespace(text="source", metadata={})]

            def chunk_documents(self, **kwargs):
                return [
                    SimpleNamespace(
                        text=f"chunk-{index}",
                        metadata={"doc_id": "no-doi", "chunk_id": index},
                    )
                    for index in range(4)
                ]

        stores = []

        class EntryVectorStore(FakeVectorStore):
            def __init__(self, **kwargs):
                super().__init__()
                self.collection_name = kwargs["collection_name"]
                stores.append(self)

            def reset_collection(self):
                self.existing.clear()
                self.add_calls.clear()

        embedders = []

        class EntryEmbedder(FakeEmbedder):
            def __init__(self, model_config, agent_configs):
                super().__init__(dimension=2)
                self.agent_embedding_profiles = {
                    "agent1": {
                        "embedding_provider": "zenmux",
                        "embedding_transport": "openai_compatible",
                    }
                }
                embedders.append(self)

            def get_model_for_agent(self, agent_name):
                return "test-embedding"

            def get_embedding_dimension(self, model):
                return 2

            async def aclose(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "load-report.json"
            persist_directory = Path(directory) / "load-chroma"
            with (
                patch.object(batch, "AgentConfig", return_value=FakeConfig()),
                patch.object(batch, "TextProcessor", FakeProcessor),
                patch.object(batch, "VectorStore", EntryVectorStore),
                patch.object(batch, "MultiModelEmbedder", EntryEmbedder),
                patch.object(batch, "setup_logging", return_value=None),
            ):
                result = batch.build_vector_databases_batch(
                    data_dir=directory,
                    literature_type_configs={
                        "test": {"path": "test", "metadata_csv": str(Path(directory) / "test.csv")}
                    },
                    agent_names=["agent1"],
                    clear_existing=True,
                    max_chunks=3,
                    persist_directory_override=str(persist_directory),
                    base_collection_name_override="load_test",
                    report_path=str(report_path),
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result["agent1"]["status"], "ok")
        self.assertEqual(result["agent1"]["newly_added"], 3)
        self.assertEqual(
            embedders[0].calls,
            [("agent1", ["chunk-0", "chunk-1"]), ("agent1", ["chunk-2"])],
        )
        self.assertEqual([len(call["ids"]) for call in stores[0].add_calls], [3])
        self.assertEqual(stores[0].collection_name, "load_test_agent1")
        self.assertEqual(report["sample_chunks"], 3)
        self.assertEqual(report["agents"]["agent1"]["logical_request_count"], 2)
        self.assertEqual(report["runtime"]["persist_directory"], str(persist_directory))


if __name__ == "__main__":
    unittest.main()
