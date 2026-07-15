import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from database.embedder import MultiModelEmbedder


def _agent_config(provider=None, api_key="test-key"):
    config = {
        "embedding_model": "text-embedding-3-small",
        "emb_url": "https://embedding.example/v1/embeddings",
        "api_key": api_key,
    }
    if provider is not None:
        config["embedding_provider"] = provider
    return config


class EmbedderProviderRoutingTests(unittest.TestCase):
    def test_configured_provider_takes_priority_for_named_agents(self):
        configs = {
            "agent1": _agent_config(" ZenMux "),
            "agent3": _agent_config("zenmux"),
        }

        embedder = MultiModelEmbedder(configs["agent1"], agent_configs=configs)

        self.assertEqual(embedder.agent_embedding_profiles["agent1"]["embedding_provider"], "zenmux")
        self.assertEqual(embedder.agent_embedding_profiles["agent3"]["embedding_provider"], "zenmux")

    def test_missing_provider_uses_agent_defaults(self):
        configs = {
            name: _agent_config()
            for name in ("agent1", "agent2", "agent3", "agent4", "custom")
        }

        embedder = MultiModelEmbedder(configs["agent1"], agent_configs=configs)

        self.assertEqual(embedder.agent_embedding_profiles["agent1"]["embedding_provider"], "openrouter")
        self.assertEqual(embedder.agent_embedding_profiles["agent2"]["embedding_provider"], "voyage")
        self.assertEqual(embedder.agent_embedding_profiles["agent3"]["embedding_provider"], "openrouter")
        self.assertEqual(embedder.agent_embedding_profiles["agent4"]["embedding_provider"], "aliyun")
        self.assertEqual(embedder.agent_embedding_profiles["custom"]["embedding_provider"], "openrouter")

    def test_default_profile_uses_model_config_provider(self):
        embedder = MultiModelEmbedder(_agent_config("zenmux"))

        self.assertEqual(embedder._get_agent_profile(None)["embedding_provider"], "zenmux")

    def test_unknown_provider_is_rejected(self):
        config = _agent_config("bailian")

        with self.assertRaisesRegex(ValueError, "Unsupported embedding provider 'bailian'"):
            MultiModelEmbedder(config, agent_configs={"agent4": config})

    def test_embed_text_routes_zenmux_and_aliyun(self):
        configs = {
            "agent1": _agent_config("zenmux"),
            "agent4": _agent_config("aliyun"),
        }
        embedder = MultiModelEmbedder(configs["agent1"], agent_configs=configs)

        with (
            patch.object(embedder, "_embed_text_openai_compatible", return_value=[1.0]) as compatible,
            patch.object(embedder, "_embed_text_aliyun", return_value=[2.0]) as aliyun,
        ):
            self.assertEqual(embedder.embed_text("alpha", agent_name="agent1"), [1.0])
            self.assertEqual(embedder.embed_text("beta", agent_name="agent4"), [2.0])

        compatible.assert_called_once_with("alpha", 3, "agent1")
        aliyun.assert_called_once_with("beta", 3, "agent4")

    def test_zenmux_missing_api_key_error_names_provider(self):
        config = _agent_config("zenmux", api_key=None)
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})

        with self.assertRaisesRegex(ValueError, "provider 'zenmux'"):
            embedder.embed_text("alpha", agent_name="agent1")


class EmbedderAsyncProviderRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_zenmux_uses_openai_compatible_async_batch(self):
        config = _agent_config("zenmux")
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})
        response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 2.0]),
                SimpleNamespace(embedding=[3.0, 4.0]),
            ]
        )
        client = Mock()
        client.embeddings.create = AsyncMock(return_value=response)

        with patch.object(embedder, "_get_async_openai_client", return_value=client):
            result = await embedder.embed_texts_openai_compatible_async(
                ["alpha", "beta"],
                agent_name="agent1",
            )

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["alpha", "beta"],
        )

    async def test_openai_compatible_batch_uses_response_indices(self):
        config = _agent_config("zenmux")
        config["embedding_transport"] = "openai_compatible"
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})
        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[1.0, 2.0]),
            ]
        )
        client = Mock()
        client.embeddings.create = AsyncMock(return_value=response)

        with patch.object(embedder, "_get_async_openai_client", return_value=client):
            result = await embedder.embed_documents_batch(["alpha", "beta"], agent_name="agent1")

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    async def test_voyage_batch_sends_all_texts_once(self):
        config = _agent_config("voyage")
        config["voyage_api_key"] = "voyage-key"
        config["embedding_model"] = "voyage-3-large"
        config["embedding_transport"] = "voyage_sdk"
        embedder = MultiModelEmbedder(config, agent_configs={"agent2": config})
        voyage_client = Mock()
        voyage_client.embed.return_value = SimpleNamespace(embeddings=[[1.0], [2.0]])

        with patch.object(embedder, "_get_voyage_client", return_value=voyage_client):
            result = await embedder.embed_documents_batch(["alpha", "beta"], agent_name="agent2")

        self.assertEqual(result, [[1.0], [2.0]])
        voyage_client.embed.assert_called_once_with(
            texts=["alpha", "beta"],
            model="voyage-3-large",
            input_type="document",
        )

    async def test_blank_batch_input_is_rejected_without_provider_call(self):
        config = _agent_config("zenmux")
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})

        with patch.object(embedder, "_get_async_openai_client") as client_factory:
            with self.assertRaisesRegex(ValueError, "blank embedding input"):
                await embedder.embed_documents_batch(["alpha", " "], agent_name="agent1")

        client_factory.assert_not_called()

    async def test_provider_failure_is_raised_instead_of_zero_vectors(self):
        config = _agent_config("zenmux")
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})
        client = Mock()
        client.embeddings.create = AsyncMock(side_effect=RuntimeError("provider unavailable"))

        with patch.object(embedder, "_get_async_openai_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await embedder.embed_documents_batch(["alpha"], agent_name="agent1")


class EmbedderClientConfigurationTests(unittest.TestCase):
    def test_openai_clients_disable_sdk_retries(self):
        config = _agent_config("zenmux")
        embedder = MultiModelEmbedder(config, agent_configs={"agent1": config})

        with patch("database.embedder.OpenAI") as openai_cls:
            embedder._get_openai_client("agent1", "key", "https://example.test/v1")
        openai_cls.assert_called_once_with(
            api_key="key",
            base_url="https://example.test/v1",
            max_retries=0,
        )

        with patch("database.embedder.AsyncOpenAI") as async_openai_cls:
            embedder._get_async_openai_client("agent1", "key", "https://example.test/v1")
        async_openai_cls.assert_called_once_with(
            api_key="key",
            base_url="https://example.test/v1",
            max_retries=0,
        )


if __name__ == "__main__":
    unittest.main()
