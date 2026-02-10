import shutil
import unittest
import uuid
from pathlib import Path

from experience import ExperienceStore


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_guideline_pack_yaml() -> str:
    # Minimal Hydra-style agent config: guidelines are embedded in agent.instructions
    # as "[Gk]. Title: content..." blocks.
    return (
        "agent:\n"
        "  instructions: |\n"
        "    When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:\n"
        "    [G0]. Format compliance: Strictly enforce STRICT JSON output format.\n"
        "    [G1]. Reduction current sign: For reduction reactions like CO2RR, predict current densities as negative values.\n"
        "    [G2]. Overpotential computation: When direct overpotentials are unavailable, compute overpotential from potentials.\n"
    )


def _make_case_pack_yaml() -> str:
    # Explicit list format supported by ExperienceStore.
    return (
        "experiences:\n"
        "  - kind: case\n"
        "    components: [\"Pt\", \"Pd\"]\n"
        "    reaction_type: OER\n"
        "    products: N/A\n"
        "    performance: \"310 mV overpotential at 10 mA/cm^2\"\n"
    )


def _make_test_tmp_dir() -> Path:
    # NOTE: On this Windows environment, `tempfile` uses `0o700` permissions which
    # results in directories that we cannot write into. Use a normal mkdir instead.
    base_dir = Path(__file__).resolve().parent
    tmp_dir = base_dir / f"_tmp_es_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    return tmp_dir


class TestExperienceStoreYaml(unittest.TestCase):
    def test_guidelines_rank_by_g_order_without_query_text(self) -> None:
        td_path = _make_test_tmp_dir()
        try:
            _write_text(td_path / "pack_guides.yaml", _make_guideline_pack_yaml())
            json_path = td_path / "experience_db.json"

            store = ExperienceStore(
                storage_path=str(json_path),
                packs_path=str(td_path),
                load_builtin_packs=False,
                guideline_search_mode="keyword",
                guideline_top_k=3,
                always_include_guidelines=True,
            )

            results = store.query_experiences(["Pt"], top_k=2)
            self.assertTrue(results, "Expected guideline experiences to be returned")
            self.assertEqual(results[0].get("kind"), "guideline")
            self.assertEqual(results[0].get("guideline_id"), "G0")
            self.assertEqual(results[1].get("guideline_id"), "G1")
        finally:
            shutil.rmtree(td_path, ignore_errors=True)

    def test_guidelines_rank_by_keywords_with_query_text(self) -> None:
        td_path = _make_test_tmp_dir()
        try:
            _write_text(td_path / "pack_guides.yaml", _make_guideline_pack_yaml())
            json_path = td_path / "experience_db.json"

            store = ExperienceStore(
                storage_path=str(json_path),
                packs_path=str(td_path),
                load_builtin_packs=False,
                guideline_search_mode="keyword",
                guideline_top_k=3,
                always_include_guidelines=True,
            )

            # Query mentions CO2RR -> guideline with "CO2RR" token should rank first (G1).
            results = store.query_experiences(["Pt"], top_k=1, query_text="Target reaction: CO2RR")
            self.assertTrue(results, "Expected guideline experiences to be returned")
            self.assertEqual(results[0].get("kind"), "guideline")
            self.assertEqual(results[0].get("guideline_id"), "G1")
        finally:
            shutil.rmtree(td_path, ignore_errors=True)

    def test_always_includes_guideline_even_with_case_match(self) -> None:
        td_path = _make_test_tmp_dir()
        try:
            _write_text(td_path / "pack_guides.yaml", _make_guideline_pack_yaml())
            _write_text(td_path / "pack_cases.yaml", _make_case_pack_yaml())
            json_path = td_path / "experience_db.json"

            store = ExperienceStore(
                storage_path=str(json_path),
                packs_path=str(td_path),
                load_builtin_packs=False,
                relevance_threshold=0.8,
                guideline_search_mode="rank",
                guideline_top_k=1,
                always_include_guidelines=True,
            )

            results = store.query_experiences(["Pt", "Pd"], top_k=2, query_text="Target reaction: OER")
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].get("kind"), "guideline")
            self.assertEqual(results[1].get("kind"), "case")
            self.assertEqual(set(results[1].get("components") or []), {"Pt", "Pd"})
        finally:
            shutil.rmtree(td_path, ignore_errors=True)
