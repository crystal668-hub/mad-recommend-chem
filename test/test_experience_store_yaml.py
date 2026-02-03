import unittest
from pathlib import Path

from experience import ExperienceStore


class TestExperienceStoreYaml(unittest.TestCase):
    def test_loads_guidelines_from_hydra_yaml_pack(self) -> None:
        repo_root = Path(__file__).resolve().parent
        packs_dir = repo_root / "experience"
        json_path = repo_root / "data" / "experience_db.json"

        store = ExperienceStore(storage_path=str(json_path), packs_path=str(packs_dir))
        results = store.query_experiences(["Pt", "Pd"], top_k=3)

        self.assertTrue(results, "Expected YAML pack experiences to be returned")
        self.assertEqual(results[0].get("kind"), "guideline")
        self.assertEqual(results[0].get("guideline_id"), "G0")
        self.assertEqual(results[0].get("title"), "Format Compliance")
        self.assertTrue(results[0].get("content"))

        self.assertTrue(
            any(str(r.get("source_file", "")).endswith("chem_perf_smoke_01_agent.yaml") for r in results),
            "Expected results sourced from chem_perf_smoke_01_agent.yaml",
        )

