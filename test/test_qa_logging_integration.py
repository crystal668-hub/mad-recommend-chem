from __future__ import annotations

import json
import logging
import shutil
import unittest
import uuid
from pathlib import Path

from qa.facade import QASystem
from qa.synthesis_state import QAResult
from utils.logger import get_run_dir, setup_logging

from test.qa_test_utils import (
    confidence_payload,
    flush_logging_handlers,
    make_base_config,
    reset_logging_state,
)


class _LoggingReactReviewedWorkflow:
    def run(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        react_logger = logging.getLogger("MAD.qa.react_reviewed")
        react_logger.info("react_reviewed_proposer_fixture_start question=%s", question)
        react_logger.warning(
            "react_reviewed_reviewer_retry role=%s cycle=%s attempt=%s error=%s",
            "search_coverage",
            1,
            1,
            "synthetic retry",
        )
        final_submission_path = artifact_root / "final_submission.json"
        final_submission_path.write_text("{}", encoding="utf-8")
        final_answer_path = artifact_root / "final_answer.md"
        final_answer_path.write_text(
            "## Direct Answer\nPt/C improves HER activity under the cited conditions.",
            encoding="utf-8",
        )
        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "workflow_mode": "react_reviewed",
                "final_answer": "## Direct Answer\nPt/C improves HER activity under the cited conditions.",
                "sections": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "content": "Pt/C improves HER activity under the cited conditions.",
                        "citation_ids": ["CIT-1"],
                        "section_confidence": confidence_payload(),
                    }
                ],
                "citations": [
                    {
                        "citation_id": "CIT-1",
                        "paper_id": "paper-1",
                        "title": "Pt/C HER in alkaline media",
                        "year": 2024,
                        "supporting_claim_ids": [],
                    }
                ],
                "claim_trace": [],
                "submission_trace": [
                    {
                        "section_id": "direct_answer",
                        "citation_ids": ["CIT-1"],
                        "step_refs": [{"trajectory_id": "traj_1", "step_number": 1}],
                        "issue_refs": [],
                    }
                ],
                "review_completion_status": "completed",
                "overall_confidence": confidence_payload(),
                "section_confidence": [
                    {
                        "section_id": "direct_answer",
                        "title": "Direct Answer",
                        "confidence": confidence_payload(),
                    }
                ],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": ["synthetic workflow warning"],
                "artifact_paths": {
                    "qa_result": str(artifact_root / "qa_result.json"),
                    "final_answer": str(final_answer_path),
                    "final_submission": str(final_submission_path),
                },
                "time_elapsed": 0.05,
            }
        )
        return result


class LoggingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"qa_logging_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        reset_logging_state()

    def tearDown(self) -> None:
        flush_logging_handlers()
        reset_logging_state()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _config(self) -> dict:
        return make_base_config(self.temp_dir, save_output=False)

    def test_setup_logging_creates_expected_run_files(self):
        config = self._config()
        setup_logging(config, run_id="logging_files")
        flush_logging_handlers()
        run_dir = Path(get_run_dir())

        self.assertTrue((self.temp_dir / "logs" / "system.log").exists())
        self.assertTrue((run_dir / "run.log").exists())
        self.assertTrue((run_dir / "events.jsonl").exists())

    def test_react_reviewed_run_emits_key_stage_logs(self):
        config = self._config()
        setup_logging(config, run_id="react_logs")
        system = QASystem(
            config=config,
            react_reviewed_workflow=_LoggingReactReviewedWorkflow(),
        )

        system.run_qa("How does Pt/C affect HER activity in 1 M KOH?")
        flush_logging_handlers()
        run_log = (Path(get_run_dir()) / "run.log").read_text(encoding="utf-8")

        self.assertIn("qa_run_start", run_log)
        self.assertIn("react_reviewed_proposer_fixture_start", run_log)
        self.assertIn("react_reviewed_reviewer_retry", run_log)
        self.assertIn("qa_run_complete", run_log)

    def test_structured_events_jsonl_contains_run_id_and_logger_name(self):
        config = self._config()
        setup_logging(config, run_id="structured_events")
        system = QASystem(
            config=config,
            react_reviewed_workflow=_LoggingReactReviewedWorkflow(),
        )

        system.run_qa("How does Pt/C affect HER activity in 1 M KOH?")
        flush_logging_handlers()
        events_path = Path(get_run_dir()) / "events.jsonl"
        rows = [read_json_line for read_json_line in events_path.read_text(encoding="utf-8").splitlines() if read_json_line.strip()]
        self.assertTrue(rows)
        parsed = [json.loads(line) for line in rows]
        qa_rows = [row for row in parsed if str(row.get("logger", "")).startswith("MAD.qa.")]
        self.assertTrue(qa_rows)
        for row in qa_rows:
            self.assertIn("ts", row)
            self.assertIn("level", row)
            self.assertIn("logger", row)
            self.assertIn("run_id", row)
            self.assertIn("msg", row)
            self.assertTrue(str(row["logger"]).startswith("MAD.qa."))

    def test_warning_path_is_logged(self):
        config = self._config()
        setup_logging(config, run_id="warning_path")
        system = QASystem(
            config=config,
            react_reviewed_workflow=_LoggingReactReviewedWorkflow(),
        )

        system.run_qa("How does Pt/C affect HER activity in 1 M KOH?")
        flush_logging_handlers()
        events_path = Path(get_run_dir()) / "events.jsonl"
        parsed = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warning_rows = [row for row in parsed if row.get("level") == "WARNING"]
        self.assertTrue(warning_rows)
        self.assertTrue(
            any("react_reviewed_reviewer_retry" in str(row.get("msg", "")) for row in warning_rows)
        )


if __name__ == "__main__":
    unittest.main()
