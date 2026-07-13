import csv
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import database.text_processor as text_processor_module
from database.text_processor import TextProcessor

try:
    import llama_index.core  # noqa: F401
except Exception:  # pragma: no cover
    llama_index_available = False
else:
    llama_index_available = True


class TextProcessorDoiExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an existing path to avoid creating new directories during tests.
        cls.processor = TextProcessor(data_dir="data/raw")

    def test_extracts_plain_doi_label(self):
        content = "# Title\n\nDOI: 10.1002/CCTC.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_doi_url(self):
        content = "Supporting information: https://doi.org/10.1002/cctc.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_dx_doi_url(self):
        content = "Link: http://dx.doi.org/10.1002/cctc.202200897\n"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_markdown_link_and_strips_punctuation(self):
        content = "See [paper](https://doi.org/10.1002/cctc.202200897)."
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_extracts_angle_bracket_url(self):
        content = "DOI: <https://doi.org/10.1002/cctc.202200897>"
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")

    def test_prefers_header_doi_over_reference_like_doi(self):
        content = (
            "# Title\n\n"
            "DOI: 10.1002/cctc.202200897\n\n"
            "## References\n"
            "- Ref A https://doi.org/10.1021/acs.jacs.0c00000\n"
        )
        doi = self.processor.extract_doi_from_content(content, "x.md", "data/raw/x.md")
        self.assertEqual(doi, "10.1002/cctc.202200897")


@unittest.skipUnless(llama_index_available, "llama-index is required for document loading tests")
class TextProcessorLiteratureTypeDocumentTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"literature_type_text_processor_{uuid.uuid4().hex[:8]}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _write_csv(self, path: Path, rows, fieldnames=None):
        resolved_fieldnames = fieldnames or ["file_name", "doi", "abstract"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_matches_pdf_name_to_markdown_and_omits_abstract_metadata(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "antiferromagnetism"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper_001.md").write_text("# Paper\n\nNo DOI here.\n", encoding="utf-8")
            csv_path = metadata_dir / "Antiferromagnetism.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "file_name": "PAPER_001.pdf",
                        "doi": "https://doi.org/10.1002/CCTC.202200897",
                        "abstract": "CSV abstract that must not reach Chroma metadata.",
                    }
                ],
            )

            docs = TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                data_dir=str(data_dir),
                metadata_csv_path=str(csv_path),
                literature_type="antiferromagnetism",
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(
                docs[0].metadata,
                {
                    "reaction_type": "antiferromagnetism",
                    "doc_id": "10.1002/cctc.202200897",
                },
            )
            self.assertNotIn("abstract", docs[0].metadata)

    def test_missing_csv_row_falls_back_to_markdown_doi(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "OER"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper_002.md").write_text(
                "# Paper\n\nDOI: 10.1021/ACS.JACS.0C00000\n",
                encoding="utf-8",
            )
            csv_path = metadata_dir / "OER.csv"
            self._write_csv(
                csv_path,
                [{"file_name": "other.pdf", "doi": "10.1002/cctc.202200897", "abstract": "Other"}],
            )

            docs = TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                data_dir=str(data_dir),
                metadata_csv_path=str(csv_path),
                literature_type="OER",
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["doc_id"], "10.1021/acs.jacs.0c00000")
            self.assertEqual(docs[0].metadata["reaction_type"], "OER")

    def test_invalid_csv_doi_falls_back_to_stable_no_doi_id(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "OER"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper_003.md").write_text("# Paper\n\nNo DOI here.\n", encoding="utf-8")
            csv_path = metadata_dir / "OER.csv"
            self._write_csv(
                csv_path,
                [{"file_name": "paper_003.pdf", "doi": "not-a-doi", "abstract": "Abstract"}],
            )

            docs = TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                data_dir=str(data_dir),
                metadata_csv_path=str(csv_path),
                literature_type="OER",
            )

            self.assertEqual(len(docs), 1)
            self.assertTrue(docs[0].metadata["doc_id"].startswith("no-doi:paper_003_"))
            self.assertEqual(docs[0].metadata["reaction_type"], "OER")

    def test_missing_required_csv_header_is_rejected(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "OER"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper.md").write_text("# Paper\n", encoding="utf-8")
            csv_path = metadata_dir / "OER.csv"
            self._write_csv(
                csv_path,
                [{"file_name": "paper.pdf", "doi": "10.1002/test"}],
                fieldnames=["file_name", "doi"],
            )

            with self.assertRaisesRegex(ValueError, "missing required columns: abstract"):
                TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                    data_dir=str(data_dir),
                    metadata_csv_path=str(csv_path),
                    literature_type="OER",
                )

    def test_duplicate_case_insensitive_pdf_basenames_are_rejected(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "OER"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper.md").write_text("# Paper\n", encoding="utf-8")
            csv_path = metadata_dir / "OER.csv"
            self._write_csv(
                csv_path,
                [
                    {"file_name": "paper.pdf", "doi": "10.1002/one", "abstract": "One"},
                    {"file_name": "PAPER.PDF", "doi": "10.1002/two", "abstract": "Two"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate file_name basename"):
                TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                    data_dir=str(data_dir),
                    metadata_csv_path=str(csv_path),
                    literature_type="OER",
                )

    def test_extra_csv_rows_are_ignored_with_warning(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "OER"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "paper.md").write_text("# Paper\n", encoding="utf-8")
            csv_path = metadata_dir / "OER.csv"
            self._write_csv(
                csv_path,
                [
                    {"file_name": "paper.pdf", "doi": "10.1002/matched", "abstract": "Matched"},
                    {"file_name": "extra.pdf", "doi": "10.1002/extra", "abstract": "Extra"},
                ],
            )

            with patch.object(text_processor_module.logger, "warning") as warning:
                docs = TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
                    data_dir=str(data_dir),
                    metadata_csv_path=str(csv_path),
                    literature_type="OER",
                )

            self.assertEqual(len(docs), 1)
            self.assertTrue(any("extra" in str(call).lower() for call in warning.call_args_list))

    def test_aggregate_loader_preserves_canonical_type_keys(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            metadata_dir = root / "metadata"
            oer_dir = data_root / "OER"
            conductivity_dir = data_root / "conductivity"
            oer_dir.mkdir(parents=True)
            conductivity_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (oer_dir / "oer.md").write_text("# OER\n", encoding="utf-8")
            (conductivity_dir / "cond.md").write_text("# Conductivity\n", encoding="utf-8")
            oer_csv = metadata_dir / "OER.csv"
            conductivity_csv = metadata_dir / "Conductivity.csv"
            self._write_csv(oer_csv, [{"file_name": "oer.pdf", "doi": "10.1002/oer", "abstract": "OER"}])
            self._write_csv(
                conductivity_csv,
                [{"file_name": "cond.pdf", "doi": "10.1002/cond", "abstract": "Conductivity"}],
            )

            docs = TextProcessor(data_dir=str(data_root)).load_literature_type_documents(
                base_dir=str(data_root),
                literature_type_configs={
                    "OER": {"path": "OER", "metadata_csv": str(oer_csv)},
                    "conductivity": {
                        "path": "conductivity",
                        "metadata_csv": str(conductivity_csv),
                    },
                },
            )

            self.assertEqual({doc.metadata["reaction_type"] for doc in docs}, {"OER", "conductivity"})


if __name__ == "__main__":
    unittest.main()

