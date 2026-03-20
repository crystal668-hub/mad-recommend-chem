from __future__ import annotations

import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from chemqa.__main__ import main as chemqa_main
from qa.synthesis_state import QAResult


def _confidence(score: float = 0.8) -> dict:
    return {
        "level": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "score": score,
        "rationale": "cli fixture",
    }


class _CapturingSystem:
    def __init__(self, *, config, config_path: str) -> None:
        self.config = config
        self.config_path = config_path

    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        workflow_mode = str((self.config.get("qa", {}) or {}).get("workflow_mode") or "ledger")
        save_output = bool((self.config.get("qa", {}) or {}).get("save_output"))
        artifact_root = Path(artifact_dir or ".")
        artifact_paths = {
            "qa_result": str(artifact_root / "qa_result.json"),
        }
        if save_output:
            artifact_paths["public_result"] = str(artifact_root / "qa_result_public.json")
        return QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "workflow_mode": workflow_mode,
                "acceptance_status": "accepted",
                "final_answer": f"workflow={workflow_mode}",
                "sections": [],
                "citations": [],
                "claim_trace": [],
                "submission_trace": [],
                "review_completion_status": "completed",
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "artifact_paths": artifact_paths,
                "time_elapsed": 0.01,
            }
        )


class ChemqaCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(".cache") / f"chemqa_cli_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_config(self, *, workflow_mode: str, save_output: bool = False) -> Path:
        path = self.temp_dir / f"config_{workflow_mode}.yaml"
        path.write_text(
            "\n".join(
                [
                    "qa:",
                    f"  workflow_mode: \"{workflow_mode}\"",
                    f"  save_output: {'true' if save_output else 'false'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _run_cli(self, argv: list[str]) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = chemqa_main(
                argv=argv,
                system_factory=_CapturingSystem,
                configure_logging=False,
            )
        self.assertEqual(0, exit_code)
        return buffer.getvalue()

    def test_cli_uses_config_workflow_mode_when_flag_is_absent(self):
        config_path = self._write_config(workflow_mode="react_reviewed")

        output = self._run_cli(
            [
                "--question",
                "How does Pt/C affect HER activity?",
                "--config",
                str(config_path),
                "--artifact-dir",
                str(self.temp_dir / "artifacts_default"),
            ]
        )

        self.assertIn("workflow=react_reviewed", output)
        self.assertIn("Saved QA artifacts:", output)

    def test_cli_workflow_mode_flag_overrides_config_to_ledger(self):
        config_path = self._write_config(workflow_mode="react_reviewed")

        output = self._run_cli(
            [
                "--question",
                "How does Pt/C affect HER activity?",
                "--config",
                str(config_path),
                "--workflow-mode",
                "ledger",
                "--artifact-dir",
                str(self.temp_dir / "artifacts_ledger"),
            ]
        )

        self.assertIn("workflow=ledger", output)

    def test_cli_workflow_mode_flag_overrides_config_to_react_reviewed(self):
        config_path = self._write_config(workflow_mode="ledger")

        output = self._run_cli(
            [
                "--question",
                "How does Pt/C affect HER activity?",
                "--config",
                str(config_path),
                "--workflow-mode",
                "react_reviewed",
                "--artifact-dir",
                str(self.temp_dir / "artifacts_react"),
            ]
        )

        self.assertIn("workflow=react_reviewed", output)

    def test_cli_workflow_mode_and_save_output_both_apply(self):
        config_path = self._write_config(workflow_mode="ledger", save_output=False)

        output = self._run_cli(
            [
                "--question",
                "How does Pt/C affect HER activity?",
                "--config",
                str(config_path),
                "--workflow-mode",
                "react_reviewed",
                "--save-output",
                "--artifact-dir",
                str(self.temp_dir / "artifacts_save_output"),
            ]
        )

        self.assertIn("workflow=react_reviewed", output)
        self.assertIn("Saved QA result:", output)


if __name__ == "__main__":
    unittest.main()
