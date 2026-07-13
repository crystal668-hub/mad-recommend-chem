import inspect
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import build_vector_db as build
except Exception:  # pragma: no cover
    build = None

from database.literature_types import LITERATURE_TYPE_CONFIGS


@unittest.skipIf(build is None, "build_vector_db dependencies are not installed")
class BuildVectorDbLiteratureTypeTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"single_literature_types_{uuid.uuid4().hex[:8]}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _fake_config(self):
        cfg = MagicMock()
        cfg.config = {
            "llm": {
                name: {
                    "model": "dummy",
                    "embedding_model": "dummy",
                    "embedding_provider": "dummy",
                }
                for name in ["agent1", "agent2", "agent3", "agent4"]
            },
            "rag": {"chunk_size": 256, "chunk_overlap": 50},
            "vector_store": {
                "persist_directory": "./data/chroma_db",
                "collection_name": "test_collection",
                "distance_metric": "cosine",
            },
        }
        cfg.get_vector_store_config.return_value = cfg.config["vector_store"]
        cfg.get_rag_config.return_value = cfg.config["rag"]
        cfg.get_llm_config.side_effect = lambda name: cfg.config["llm"][name]
        return cfg

    def test_builder_uses_unified_literature_type_loader(self):
        with self._tempdir() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            processor = MagicMock()
            processor.load_literature_type_documents.return_value = []

            with patch.object(build, "AgentConfig", return_value=self._fake_config()), patch.object(
                build, "TextProcessor", return_value=processor
            ), patch.object(build, "setup_logging", return_value=None):
                build.build_vector_database(
                    data_dir=str(data_dir),
                    literature_type_configs=LITERATURE_TYPE_CONFIGS,
                    agent_name="agent1",
                )

            processor.load_literature_type_documents.assert_called_once_with(
                base_dir=str(data_dir),
                literature_type_configs=LITERATURE_TYPE_CONFIGS,
            )

    def test_builder_exposes_only_unified_ingestion_config(self):
        parameters = inspect.signature(build.build_vector_database).parameters

        self.assertIn("literature_type_configs", parameters)
        self.assertNotIn("reaction_configs", parameters)
        self.assertIs(build.LITERATURE_TYPE_CONFIGS, LITERATURE_TYPE_CONFIGS)


if __name__ == "__main__":
    unittest.main()
