from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import pymupdf as fitz


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skill"
EXPECTED_SKILLS = {
    "paper-retrieval": "paper_retrieval.py",
    "paper-access": "paper_access.py",
    "paper-parse": "paper_parse.py",
    "paper-rerank": "paper_rerank.py",
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in pages:
            page = document.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 555, 800), page_text, fontsize=11)
        return document.tobytes()
    finally:
        document.close()


class SkillBundleTests(unittest.TestCase):
    def test_expected_skill_bundles_exist_with_required_files(self):
        for skill_name, script_name in EXPECTED_SKILLS.items():
            bundle_dir = SKILL_ROOT / skill_name
            self.assertTrue(bundle_dir.exists(), str(bundle_dir))
            self.assertTrue((bundle_dir / "SKILL.md").exists(), str(bundle_dir / "SKILL.md"))
            self.assertTrue((bundle_dir / "scripts" / script_name).exists(), str(bundle_dir / "scripts" / script_name))
            self.assertTrue((bundle_dir / "references").exists(), str(bundle_dir / "references"))

    def test_skill_frontmatter_uses_searchable_metadata(self):
        for skill_name in EXPECTED_SKILLS:
            text = (SKILL_ROOT / skill_name / "SKILL.md").read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), skill_name)
            self.assertIn("name:", text, skill_name)
            self.assertIn("description: Use when", text, skill_name)

    def test_scripts_do_not_import_repo_runtime_modules(self):
        forbidden_imports = ("from qa", "import qa", "from agents", "import agents", "from utils", "import utils")
        for skill_name, script_name in EXPECTED_SKILLS.items():
            script_text = (SKILL_ROOT / skill_name / "scripts" / script_name).read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                self.assertNotIn(forbidden, script_text, f"{skill_name} leaked repo dependency via {forbidden}")

    def test_paper_parse_defaults_to_pymupdf_with_docling_fallback(self):
        module = _load_module(
            f"paper_parse_{uuid.uuid4().hex}",
            SKILL_ROOT / "paper-parse" / "scripts" / "paper_parse.py",
        )

        config = module.ParserConfig()
        self.assertEqual("pymupdf", config.primary_backend)
        self.assertEqual("docling", config.secondary_backend)

    def test_paper_parse_uses_docling_when_pymupdf_output_is_rejected(self):
        module = _load_module(
            f"paper_parse_{uuid.uuid4().hex}",
            SKILL_ROOT / "paper-parse" / "scripts" / "paper_parse.py",
        )
        engine = module.PaperParseEngine(config=module.ParserConfig())
        rejected_attempt = module.ExtractionAttempt(
            extractor="pymupdf",
            succeeded=True,
            fulltext="1",
            page_texts=["1", "2", "3"],
            sections=[],
            page_count=3,
            metrics={"total_chars": 3, "reasons": ["total_chars_below_threshold"]},
            usable=False,
        )
        accepted_attempt = module.ExtractionAttempt(
            extractor="docling",
            succeeded=True,
            fulltext="Results\n" + ("Useful text. " * 200),
            page_texts=["Useful text. " * 100] * 3,
            sections=[],
            page_count=3,
            metrics={"total_chars": 2400, "reasons": []},
            usable=True,
        )
        engine._extract_with_pymupdf = lambda pdf_bytes: rejected_attempt  # type: ignore[method-assign]
        engine._extract_with_docling = lambda pdf_path: accepted_attempt  # type: ignore[method-assign]

        tmpdir = Path(tempfile.mkdtemp(prefix="skill_parse_"))
        try:
            result = engine.process_pdf_bytes(
                document_id="paper-1",
                pdf_bytes=_make_pdf_bytes(["1", "2", "3"]),
                output_dir=tmpdir,
            )
            self.assertEqual("docling", result["extractor"])
            self.assertEqual("fulltext_indexed", result["fulltext_status"])
            self.assertTrue(Path(result["fulltext_artifact_path"]).exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
