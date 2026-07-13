import inspect
import io
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import build_vector_db_batch as batch
except Exception:  # pragma: no cover
    batch = None

from database.literature_types import LITERATURE_TYPE_CONFIGS


@unittest.skipIf(batch is None, "build_vector_db_batch dependencies are not installed")
class BuildVectorDbBatchLiteratureTypeTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"batch_literature_types_{uuid.uuid4().hex[:8]}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _fake_config(self):
        cfg = MagicMock()
        cfg.config = {
            "llm": {
                "agent1": {
                    "embedding_model": "dummy",
                    "embedding_provider": "dummy",
                }
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
        cfg.get_llm_config.return_value = cfg.config["llm"]["agent1"]
        return cfg

    def test_builder_uses_unified_literature_type_loader(self):
        with self._tempdir() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            processor = MagicMock()
            processor.load_literature_type_documents.return_value = ["doc"]
            processor.chunk_documents.return_value = []

            with patch.object(batch, "AgentConfig", return_value=self._fake_config()), patch.object(
                batch, "TextProcessor", return_value=processor
            ), patch.object(batch, "setup_logging", return_value=None):
                batch.build_vector_databases_batch(
                    data_dir=str(data_dir),
                    literature_type_configs=LITERATURE_TYPE_CONFIGS,
                    agent_names=["agent1"],
                )

            processor.load_literature_type_documents.assert_called_once_with(
                base_dir=str(data_dir),
                literature_type_configs=LITERATURE_TYPE_CONFIGS,
            )

    def test_builder_exposes_only_unified_ingestion_config(self):
        parameters = inspect.signature(batch.build_vector_databases_batch).parameters

        self.assertIn("literature_type_configs", parameters)
        self.assertNotIn("input_layout", parameters)
        self.assertNotIn("reaction_configs", parameters)
        self.assertNotIn("category_configs", parameters)
        self.assertIs(batch.LITERATURE_TYPE_CONFIGS, LITERATURE_TYPE_CONFIGS)

    def test_cli_help_does_not_advertise_input_layout(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["build_vector_db_batch.py", "--help"]), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                batch.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertNotIn("--input-layout", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
