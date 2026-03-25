from __future__ import annotations

import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from qa.chembench_smoke import (
    ChembenchSmokeCase,
    extract_short_answer,
    load_smoke_cases,
    main as chembench_smoke_main,
    run_smoke_case,
    score_case_answer,
)
from qa.synthesis_state import QAResult
from utils import ensure_dir, load_json, save_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _confidence(score: float = 0.8) -> dict:
    return {
        "level": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "score": score,
        "rationale": "chembench smoke fixture",
    }


class _StubSmokeSystem:
    response_by_question: dict[str, str] = {}
    workflow_mode = "react_reviewed"
    review_completion_status = "completed"
    reasoning_reviewer_status = "completed"
    citations = [{"citation_id": "CIT-1", "paper_id": "paper-1", "title": "fixture", "year": 2024}]
    captured_calls: list[dict] = []

    def __init__(self, *, config, config_path: str) -> None:
        self.config = config
        self.config_path = config_path

    def run_qa(self, *, question: str, context=None, artifact_dir=None):
        artifact_root = ensure_dir(Path(artifact_dir or "."))
        self.__class__.captured_calls.append(
            {
                "question": question,
                "context": context,
                "artifact_dir": str(artifact_root),
            }
        )
        save_json(
            [{"reviewer_role": "reasoning_consistency", "status": self.__class__.reasoning_reviewer_status}],
            artifact_root / "review_statuses.json",
        )
        qa_result_path = artifact_root / "qa_result.json"
        result = QAResult.model_validate(
            {
                "question": question,
                "language": "en",
                "workflow_mode": self.__class__.workflow_mode,
                "acceptance_status": "accepted",
                "final_answer": self.__class__.response_by_question[question],
                "sections": [],
                "citations": list(self.__class__.citations),
                "claim_trace": [],
                "submission_trace": [],
                "review_completion_status": self.__class__.review_completion_status,
                "overall_confidence": _confidence(),
                "section_confidence": [],
                "insufficient_evidence": False,
                "limitations_summary": "",
                "retrieval_diagnostics_summary": "",
                "execution_warnings": [],
                "artifact_paths": {
                    "qa_result": str(qa_result_path),
                },
                "time_elapsed": 0.01,
            }
        )
        save_json(result.model_dump(exclude_none=True), qa_result_path)
        return result


class ChembenchSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = REPO_ROOT / ".cache" / f"chembench_smoke_{uuid.uuid4().hex[:8]}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.temp_dir / "config.yaml"
        self.config_path.write_text("qa:\n  workflow_mode: react_reviewed\n", encoding="utf-8")
        _StubSmokeSystem.response_by_question = {}
        _StubSmokeSystem.workflow_mode = "react_reviewed"
        _StubSmokeSystem.review_completion_status = "completed"
        _StubSmokeSystem.reasoning_reviewer_status = "completed"
        _StubSmokeSystem.citations = [{"citation_id": "CIT-1", "paper_id": "paper-1", "title": "fixture", "year": 2024}]
        _StubSmokeSystem.captured_calls = []

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_smoke_cases_reads_repo_fixture(self):
        cases = load_smoke_cases(str(REPO_ROOT / "evals" / "chembench_smoke_cases.yaml"))

        self.assertEqual(3, len(cases))
        self.assertEqual("oxidation_states", cases[0].name)
        self.assertEqual("integer", cases[0].type)
        self.assertEqual("4e846028-9f8f-44f2-eb3e-59316479df10", cases[1].uuid)

    def test_extract_short_answer_prefers_explicit_marker(self):
        answer, extracted_ok, source = extract_short_answer(
            "Reasoning here.\nFINAL_SHORT_ANSWER: 174\n"
        )

        self.assertEqual("174", answer)
        self.assertTrue(extracted_ok)
        self.assertEqual("final_short_answer_marker", source)

    def test_score_case_answer_integer_uses_last_integer(self):
        case = ChembenchSmokeCase.model_validate(
            {
                "subset": "analytical_chemistry",
                "name": "molecular_structure",
                "uuid": "fixture",
                "question": "How many modes?",
                "gold": "174",
                "type": "integer",
                "keywords": [],
                "source_url": "https://example.com",
            }
        )

        correct, normalized_prediction, normalized_gold = score_case_answer(
            case=case,
            predicted_answer="3N-6 gives 174",
        )

        self.assertTrue(correct)
        self.assertEqual("174", normalized_prediction)
        self.assertEqual("174", normalized_gold)

    def test_run_smoke_case_uses_marker_and_reads_reasoning_status(self):
        case = ChembenchSmokeCase.model_validate(
            {
                "subset": "analytical_chemistry",
                "name": "analytical_chemistry_3",
                "uuid": "fixture-choice",
                "question": "Which method fits trace multi-element quantification?",
                "gold": "Inductively coupled plasma optical emission spectroscopy",
                "type": "choice_text",
                "keywords": ["ICP-OES"],
                "source_url": "https://example.com",
            }
        )
        _StubSmokeSystem.response_by_question = {
            case.question: (
                "Use an elemental analysis method with simultaneous multi-element coverage.\n"
                "FINAL_SHORT_ANSWER: ICP-OES"
            )
        }

        report = run_smoke_case(
            case,
            config={"qa": {"workflow_mode": "react_reviewed"}},
            config_path=str(self.config_path),
            artifact_dir=str(self.temp_dir / "artifacts_case"),
            system_factory=_StubSmokeSystem,
        )

        self.assertTrue(report.pipeline_ok)
        self.assertTrue(report.extracted_ok)
        self.assertTrue(report.correct)
        self.assertEqual("completed", report.reasoning_reviewer_status)
        self.assertEqual("ICP-OES", report.predicted_answer)
        self.assertTrue((Path(report.artifact_dir) / "chembench_smoke_case_report.json").exists())

    def test_main_filters_selected_case_and_writes_suite_report(self):
        selected_case = load_smoke_cases(str(REPO_ROOT / "evals" / "chembench_smoke_cases.yaml"))[1]
        _StubSmokeSystem.response_by_question = {
            selected_case.question: f"FINAL_SHORT_ANSWER: {selected_case.gold}",
        }
        _StubSmokeSystem.citations = []

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            exit_code = chembench_smoke_main(
                argv=[
                    "--config",
                    str(self.config_path),
                    "--cases-file",
                    str(REPO_ROOT / "evals" / "chembench_smoke_cases.yaml"),
                    "--case",
                    selected_case.name,
                    "--artifact-root",
                    str(self.temp_dir / "suite_run"),
                ],
                system_factory=_StubSmokeSystem,
                configure_logging=False,
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(_StubSmokeSystem.captured_calls))
        self.assertIn("FINAL_SHORT_ANSWER", _StubSmokeSystem.captured_calls[0]["context"])
        suite_report = load_json(self.temp_dir / "suite_run" / "chembench_smoke_report.json")
        self.assertEqual(1, suite_report["summary"]["total"])
        self.assertEqual(1, suite_report["summary"]["correct"])
        self.assertIn("ChemBench smoke report:", output_buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
