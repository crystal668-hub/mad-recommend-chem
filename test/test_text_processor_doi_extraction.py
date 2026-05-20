import unittest
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from database.text_processor import TextProcessor

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

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


@unittest.skipIf(pd is None, "pandas is required for flat XLSX metadata tests")
@unittest.skipUnless(llama_index_available, "llama-index is required for document loading tests")
class TextProcessorFlatDocumentTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"flat_text_processor_{uuid.uuid4().hex[:8]}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _write_xlsx(self, path: Path, rows):
        pd.DataFrame(rows).to_excel(path, index=False)

    def test_loads_flat_document_doi_from_xlsx_id(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            (data_dir / "paper_001.md").write_text("# Paper\n\nNo DOI here.\n", encoding="utf-8")
            xlsx_path = metadata_dir / "metadata.xlsx"
            self._write_xlsx(
                xlsx_path,
                [{" ID ": "paper_001", " DOI ": "https://doi.org/10.1002/CCTC.202200897"}],
            )

            docs = TextProcessor(data_dir=str(data_dir)).load_flat_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["doc_id"], "10.1002/cctc.202200897")
            self.assertEqual(docs[0].metadata["reaction_type"], "Antiferromagnetism")

    def test_flat_document_falls_back_to_markdown_doi_when_id_missing(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            (data_dir / "paper_002.md").write_text(
                "# Paper\n\nDOI: 10.1021/ACS.JACS.0C00000\n",
                encoding="utf-8",
            )
            xlsx_path = metadata_dir / "metadata.xlsx"
            self._write_xlsx(xlsx_path, [{"id": "other", "doi": "10.1002/cctc.202200897"}])

            docs = TextProcessor(data_dir=str(data_dir)).load_flat_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["doc_id"], "10.1021/acs.jacs.0c00000")
            self.assertEqual(docs[0].metadata["reaction_type"], "Antiferromagnetism")

    def test_flat_document_falls_back_to_stable_no_doi_id(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            (data_dir / "paper_003.md").write_text("# Paper\n\nNo DOI here.\n", encoding="utf-8")
            xlsx_path = metadata_dir / "metadata.xlsx"
            self._write_xlsx(xlsx_path, [{"id": "paper_003", "doi": "not-a-doi"}])

            docs = TextProcessor(data_dir=str(data_dir)).load_flat_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
            )

            self.assertEqual(len(docs), 1)
            self.assertTrue(docs[0].metadata["doc_id"].startswith("no-doi:paper_003_"))
            self.assertEqual(docs[0].metadata["reaction_type"], "Antiferromagnetism")

    def test_flat_document_allows_custom_reaction_type(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            metadata_dir = root / "metadata"
            data_dir.mkdir()
            metadata_dir.mkdir()
            (data_dir / "paper_004.md").write_text("# Paper\n\nNo DOI here.\n", encoding="utf-8")
            xlsx_path = metadata_dir / "metadata.xlsx"
            self._write_xlsx(xlsx_path, [{"id": "paper_004", "doi": "10.1002/cctc.202200897"}])

            docs = TextProcessor(data_dir=str(data_dir)).load_flat_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
                reaction_type="CustomType",
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["reaction_type"], "CustomType")


@unittest.skipIf(pd is None, "pandas is required for category XLSX metadata tests")
@unittest.skipUnless(llama_index_available, "llama-index is required for document loading tests")
class TextProcessorCategoryDocumentTests(unittest.TestCase):
    @contextmanager
    def _tempdir(self):
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"category_text_processor_{uuid.uuid4().hex[:8]}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _write_xlsx(self, path: Path, rows):
        pd.DataFrame(rows).to_excel(path, index=False)

    def test_loads_category_document_doi_from_best_matching_xlsx_column(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "conductivity"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "1.md").write_text("# Paper 1\n\nNo DOI here.\n", encoding="utf-8")
            (data_dir / "2.md").write_text("# Paper 2\n\nNo DOI here.\n", encoding="utf-8")
            xlsx_path = metadata_dir / "Conductivity.xlsx"
            self._write_xlsx(
                xlsx_path,
                [
                    {
                        "row_number": "300",
                        "doi": "10.1002/cctc.202200897",
                        "Unnamed: 10": "1",
                    },
                    {
                        "row_number": "301",
                        "doi": "10.1021/acs.jacs.0c00000",
                        "Unnamed: 10": "2",
                    },
                ],
            )

            docs = TextProcessor(data_dir=str(root / "data")).load_category_directory_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
                category_label="Conductivity",
            )

            doc_ids = {doc.metadata["doc_id"] for doc in docs}
            self.assertEqual(len(docs), 2)
            self.assertEqual(
                doc_ids,
                {"10.1002/cctc.202200897", "10.1021/acs.jacs.0c00000"},
            )
            self.assertTrue(all(doc.metadata["reaction_type"] == "conductivity" for doc in docs))

    def test_category_document_missing_metadata_falls_back_to_stable_no_doi_id(self):
        with self._tempdir() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "conductivity"
            metadata_dir = root / "metadata"
            data_dir.mkdir(parents=True)
            metadata_dir.mkdir()
            (data_dir / "1.md").write_text("# Paper 1\n\nNo DOI here.\n", encoding="utf-8")
            (data_dir / "2.md").write_text("# Paper 2\n\nNo DOI here.\n", encoding="utf-8")
            xlsx_path = metadata_dir / "Conductivity.xlsx"
            self._write_xlsx(
                xlsx_path,
                [{"doi": "10.1002/cctc.202200897", "Unnamed: 10": "1"}],
            )

            docs = TextProcessor(data_dir=str(root / "data")).load_category_directory_documents(
                data_dir=str(data_dir),
                metadata_xlsx_path=str(xlsx_path),
                category_label="Conductivity",
            )

            doc_ids = {doc.metadata["doc_id"] for doc in docs}
            self.assertEqual(len(docs), 2)
            self.assertIn("10.1002/cctc.202200897", doc_ids)
            self.assertTrue(any(doc_id.startswith("no-doi:2_") for doc_id in doc_ids))
            self.assertTrue(all(doc.metadata["reaction_type"] == "conductivity" for doc in docs))


if __name__ == "__main__":
    unittest.main()

