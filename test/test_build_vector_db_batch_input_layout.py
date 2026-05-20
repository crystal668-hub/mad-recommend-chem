import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import build_vector_db_batch as batch
except Exception:  # pragma: no cover
    batch = None


@unittest.skipIf(batch is None, "build_vector_db_batch dependencies are not installed")
class BuildVectorDbBatchInputLayoutTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"batch_input_layout_{uuid.uuid4().hex[:8]}"
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

    def test_reaction_layout_uses_reaction_loader(self):
        with self._tempdir() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            processor = MagicMock()
            processor.load_reaction_documents.return_value = ["doc"]
            processor.chunk_documents.return_value = []

            with patch.object(batch, "AgentConfig", return_value=self._fake_config()), patch.object(
                batch, "TextProcessor", return_value=processor
            ), patch.object(batch, "setup_logging", return_value=None):
                batch.build_vector_databases_batch(
                    data_dir=str(data_dir),
                    input_layout="reaction",
                    agent_names=["agent1"],
                )

            processor.load_reaction_documents.assert_called_once()
            processor.load_flat_documents.assert_not_called()
            processor.load_category_documents.assert_not_called()

    def test_flat_layout_uses_flat_loader(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            xlsx_path = metadata_dir / "metadata.xlsx"
            xlsx_path.write_bytes(b"placeholder")

            processor = MagicMock()
            processor.load_flat_documents.return_value = ["doc"]
            processor.chunk_documents.return_value = []

            with patch.object(batch, "AgentConfig", return_value=self._fake_config()), patch.object(
                batch, "TextProcessor", return_value=processor
            ), patch.object(batch, "setup_logging", return_value=None):
                batch.build_vector_databases_batch(
                    data_dir=str(data_dir),
                    input_layout="flat",
                    metadata_xlsx_path=str(xlsx_path),
                    agent_names=["agent1"],
                )

            processor.load_flat_documents.assert_called_once_with(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
                reaction_type="Antiferromagnetism",
            )
            processor.load_reaction_documents.assert_not_called()
            processor.load_category_documents.assert_not_called()

    def test_category_layout_uses_category_loader(self):
        with self._tempdir() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            processor = MagicMock()
            processor.load_category_documents.return_value = ["doc"]
            processor.chunk_documents.return_value = []

            with patch.object(batch, "AgentConfig", return_value=self._fake_config()), patch.object(
                batch, "TextProcessor", return_value=processor
            ), patch.object(batch, "setup_logging", return_value=None):
                batch.build_vector_databases_batch(
                    data_dir=str(data_dir),
                    input_layout="category",
                    agent_names=["agent1"],
                )

            processor.load_category_documents.assert_called_once_with(
                base_dir=str(data_dir),
                category_configs=batch.CATEGORY_CONFIGS,
            )
            processor.load_reaction_documents.assert_not_called()
            processor.load_flat_documents.assert_not_called()

    def test_flat_metadata_auto_resolve_requires_exactly_one_xlsx(self):
        with self._tempdir() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            with patch.object(batch, "Path", side_effect=lambda value=".": Path(tmp) / value):
                with self.assertRaisesRegex(ValueError, "exactly one metadata XLSX"):
                    batch._resolve_metadata_xlsx_path(None)


if __name__ == "__main__":
    unittest.main()
