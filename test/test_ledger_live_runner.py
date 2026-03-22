from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from qa.ledger_live_runner import (
    _write_shadow_config,
    analyze_artifact_dir,
    analyze_react_control,
)
from utils import save_json


class LedgerLiveRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"ledger_live_runner_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_shadow_config_forces_ledger_mode(self) -> None:
        shadow_path = self.temp_dir / "shadow.yaml"
        _write_shadow_config(shadow_path)
        text = shadow_path.read_text(encoding="utf-8")

        self.assertIn("workflow_mode: ledger", text)
        self.assertIn("save_output: false", text)

    def test_analyze_artifact_dir_reports_stall_after_document_acquisition(self) -> None:
        artifact_dir = self.temp_dir / "artifact"
        (artifact_dir / "entity_resolver").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "fulltext").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "indices").mkdir(parents=True, exist_ok=True)

        save_json({"qa": {"workflow_mode": "ledger"}}, artifact_dir / "runtime_manifest.json")
        save_json(
            {
                "entities": [
                    {
                        "mention": "Pt/C",
                        "canonical_name": "platinum on carbon",
                        "query_anchors": ["Pt/C", "platinum on carbon"],
                    }
                ],
                "condition_mentions": [{"raw_value": "1 M KOH", "normalized_value": "1 M KOH"}],
                "unresolved_mentions": [{"mention": "activity"}],
            },
            artifact_dir / "entity_resolver" / "entity_pack.json",
        )
        save_json(
            {
                "display_name": "Ethanol Fuel Cell Electrocatalysts and Battery Interfaces",
                "abstract": "A broad review of ethanol oxidation, fuel cells, and battery interfaces.",
            },
            artifact_dir / "provider_raw" / "openalex" / "paper_a.json",
        )
        save_json({"paper_id": "paper_a"}, artifact_dir / "indices" / "paper_a.json")
        (artifact_dir / "fulltext" / "paper_a.fulltext.txt").write_text("dummy", encoding="utf-8")

        diagnosis = analyze_artifact_dir(
            question="How does Pt/C affect HER activity in 1 M KOH?",
            artifact_dir=artifact_dir,
        )

        self.assertEqual("stalled_after_document_acquisition", diagnosis["run_state"])
        self.assertEqual("ledger", diagnosis["workflow_mode"])
        self.assertGreaterEqual(diagnosis["grounding"]["resolved_entity_count"], 1)
        self.assertIn(
            "Run reached document acquisition but did not materialize an evidence ledger.",
            diagnosis["synthesis"]["notes"],
        )
        self.assertTrue(diagnosis["retrieval"]["off_topic_titles"])

    def test_analyze_react_control_flags_non_ledger_drift(self) -> None:
        control_path = self.temp_dir / "react_control.json"
        save_json(
            {
                "workflow_mode": "react_reviewed",
                "review_completion_status": "incomplete",
                "final_answer": "Iridium catalysts improve acidic OER performance.",
                "execution_warnings": ["reviewer invalid_json output"],
            },
            control_path,
        )

        diagnosis = analyze_react_control(control_path)

        self.assertEqual("react_reviewed", diagnosis["workflow_mode"])
        self.assertEqual("incomplete", diagnosis["review_completion_status"])
        self.assertTrue(any("not a ledger run" in item for item in diagnosis["flags"]))
        self.assertTrue(any("Ir/OER" in item for item in diagnosis["flags"]))


if __name__ == "__main__":
    unittest.main()
