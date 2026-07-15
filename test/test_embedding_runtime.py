import asyncio
import math
import unittest

from database.embedding_runtime import (
    EmbeddingFailure,
    EmbeddingQuotaScheduler,
    EmbeddingRuntimeSettings,
    EmbeddingValidationError,
    QuotaPolicy,
    RetryPolicy,
    build_text_batches,
    validate_embeddings,
)


class EmbeddingRuntimeSettingsTests(unittest.TestCase):
    def test_defaults_and_legacy_overrides(self):
        settings = EmbeddingRuntimeSettings.from_config(
            {},
            agent_configs={
                "agent1": {
                    "embedding_provider": "zenmux",
                    "embedding_quota_group": "myrimate",
                }
            },
            legacy_batch_size=24,
            legacy_concurrency=3,
        )

        self.assertEqual(settings.request_batch_size_for("agent1"), 24)
        self.assertEqual(settings.quota_groups["myrimate"].initial_inflight, 3)
        self.assertEqual(settings.quota_groups["myrimate"].max_inflight, 3)

    def test_unknown_quota_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown quota group"):
            EmbeddingRuntimeSettings.from_config(
                {"quota_groups": {"known": {"max_inflight": 1}}},
                agent_configs={
                    "agent1": {
                        "embedding_quota_group": "missing",
                    }
                },
            )


class EmbeddingBatchTests(unittest.TestCase):
    def test_batches_respect_item_and_token_limits(self):
        batches = build_text_batches(
            agent_name="agent1",
            texts=["a" * 8, "b" * 8, "c" * 8, "d" * 8, "e" * 8],
            ids=["1", "2", "3", "4", "5"],
            metadatas=[{} for _ in range(5)],
            max_items=3,
            max_tokens=4,
            token_estimator=lambda text: len(text) // 4,
        )

        self.assertEqual([batch.ids for batch in batches], [["1", "2"], ["3", "4"], ["5"]])
        self.assertTrue(all(batch.estimated_tokens <= 4 for batch in batches))

    def test_blank_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "blank embedding input"):
            build_text_batches(
                agent_name="agent1",
                texts=["ok", "   "],
                ids=["1", "2"],
                metadatas=[{}, {}],
                max_items=10,
            )


class EmbeddingValidationTests(unittest.TestCase):
    def test_valid_vectors_are_normalized_to_float_lists(self):
        result = validate_embeddings([[1, 2], [3.0, 4.0]], expected_count=2, expected_dimension=2)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_invalid_count_dimension_values_and_zero_vectors_are_rejected(self):
        cases = [
            ([[1.0, 2.0]], 2, 2),
            ([[1.0]], 1, 2),
            ([[math.nan, 1.0]], 1, 2),
            ([[math.inf, 1.0]], 1, 2),
            ([[0.0, 0.0]], 1, 2),
        ]
        for vectors, count, dimension in cases:
            with self.subTest(vectors=vectors):
                with self.assertRaises(EmbeddingValidationError):
                    validate_embeddings(vectors, expected_count=count, expected_dimension=dimension)

    def test_failure_record_is_sanitized(self):
        failure = EmbeddingFailure(
            agent="agent1",
            collection="collection_agent1",
            chunk_id="chunk-1",
            provider="zenmux",
            model="model",
            quota_group="myrimate",
            attempts=2,
            error_type="rate_limit",
            retryable=True,
            message="Authorization: Bearer secret-token\n" + "x" * 1000,
        )
        payload = failure.to_dict()
        self.assertNotIn("secret-token", payload["message"])
        self.assertLessEqual(len(payload["message"]), 500)
        self.assertNotIn("text", payload)


class EmbeddingQuotaSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_and_global_caps_cover_multiple_agents(self):
        scheduler = EmbeddingQuotaScheduler(
            global_max_inflight=2,
            quota_groups={
                "shared": QuotaPolicy(initial_inflight=2, max_inflight=2),
            },
            retry_policy=RetryPolicy(max_attempts=1),
        )
        lock = asyncio.Lock()
        inflight = 0
        peak = 0

        async def operation():
            nonlocal inflight, peak
            async with lock:
                inflight += 1
                peak = max(peak, inflight)
            await asyncio.sleep(0.02)
            async with lock:
                inflight -= 1
            return "ok"

        results = await asyncio.gather(
            *[
                scheduler.execute("shared", estimated_tokens=10, operation=operation)
                for _ in range(8)
            ]
        )

        self.assertEqual(results, ["ok"] * 8)
        self.assertEqual(peak, 2)
        self.assertEqual(scheduler.snapshot()["shared"]["peak_inflight"], 2)

    async def test_retry_uses_one_attempt_per_slot_and_stops_on_permanent_error(self):
        scheduler = EmbeddingQuotaScheduler(
            global_max_inflight=1,
            quota_groups={"q": QuotaPolicy(initial_inflight=1, max_inflight=1)},
            retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0),
        )
        attempts = 0

        class RetryableError(RuntimeError):
            status_code = 503

        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RetryableError("temporary")
            return "done"

        self.assertEqual(await scheduler.execute("q", 1, flaky), "done")
        self.assertEqual(attempts, 3)
        self.assertEqual(scheduler.snapshot()["q"]["attempts"], 3)

        permanent_attempts = 0

        class PermanentError(RuntimeError):
            status_code = 400

        async def permanent():
            nonlocal permanent_attempts
            permanent_attempts += 1
            raise PermanentError("bad request")

        with self.assertRaises(PermanentError):
            await scheduler.execute("q", 1, permanent)
        self.assertEqual(permanent_attempts, 1)

    async def test_throttle_halves_effective_concurrency(self):
        scheduler = EmbeddingQuotaScheduler(
            global_max_inflight=4,
            quota_groups={"q": QuotaPolicy(initial_inflight=4, max_inflight=4, cooldown_seconds=0)},
            retry_policy=RetryPolicy(max_attempts=1),
        )

        class ThrottleError(RuntimeError):
            status_code = 429

        async def throttled():
            raise ThrottleError("slow down")

        with self.assertRaises(ThrottleError):
            await scheduler.execute("q", 1, throttled)

        self.assertEqual(scheduler.snapshot()["q"]["effective_inflight"], 2)


if __name__ == "__main__":
    unittest.main()
