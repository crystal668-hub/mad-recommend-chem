from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHEMQA_REVIEW_ROOT = REPO_ROOT / "skill" / "chemqa-review"
DEBATECLAW_SKILL_ROOT = REPO_ROOT.parent / "debateclaw-v1" / "skill"
LOCAL_SKILL_ROOT = REPO_ROOT / "skill"
REQUIRED_LOCAL_SKILLS = (
    "paper-retrieval",
    "paper-access",
    "paper-parse",
    "paper-rerank",
)


def _run_json(command: list[str]) -> dict:
    return _run_json_with_env(command, env=None)


def _run_json_with_env(command: list[str], env: dict[str, str] | None) -> dict:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic only
        raise AssertionError(
            f"Command did not emit JSON: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc


class ChemQAReviewBundleTests(unittest.TestCase):
    def _copy_skill(self, skills_root: Path, skill_name: str) -> None:
        source = LOCAL_SKILL_ROOT / skill_name
        target = skills_root / skill_name
        shutil.copytree(source, target)

    def _install_skills(
        self,
        *,
        include_engine: bool = True,
        include_local_skills: bool = True,
    ) -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="chemqa_review_skill_"))
        skills_root = temp_root / "skills"
        skills_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CHEMQA_REVIEW_ROOT, skills_root / "chemqa-review")
        if include_engine:
            shutil.copytree(DEBATECLAW_SKILL_ROOT, skills_root / "debateclaw-v1")
        if include_local_skills:
            for skill_name in REQUIRED_LOCAL_SKILLS:
                self._copy_skill(skills_root, skill_name)
        return skills_root

    def test_bundle_contains_required_files(self):
        required_paths = (
            CHEMQA_REVIEW_ROOT / "SKILL.md",
            CHEMQA_REVIEW_ROOT / "scripts" / "check_runtime.py",
            CHEMQA_REVIEW_ROOT / "scripts" / "compile_runplan.py",
            CHEMQA_REVIEW_ROOT / "scripts" / "materialize_runplan.py",
            CHEMQA_REVIEW_ROOT / "scripts" / "launch_from_preset.py",
            CHEMQA_REVIEW_ROOT / "scripts" / "collect_artifacts.py",
            CHEMQA_REVIEW_ROOT / "workflows" / "chemqa-review@1.json",
            CHEMQA_REVIEW_ROOT / "presets" / "chemqa-review@1.json",
            CHEMQA_REVIEW_ROOT / "control" / "model-profiles" / "chemqa-review-default.json",
            CHEMQA_REVIEW_ROOT / "control" / "config-snapshots" / "react-reviewed-default.json",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "coordinator.md",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "proposer-main.md",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "reviewer-search-coverage.md",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "reviewer-evidence-trace.md",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "reviewer-reasoning-consistency.md",
            CHEMQA_REVIEW_ROOT / "prompts" / "contracts" / "reviewer-counterevidence.md",
        )
        for path in required_paths:
            self.assertTrue(path.exists(), str(path))

    def test_scripts_do_not_import_repo_runtime_modules(self):
        forbidden_imports = ("from qa", "import qa", "from agents", "import agents", "from utils", "import utils")
        for script_path in (CHEMQA_REVIEW_ROOT / "scripts").glob("*.py"):
            script_text = script_path.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                self.assertNotIn(forbidden, script_text, f"{script_path.name} leaked repo dependency via {forbidden}")

    def test_check_runtime_fails_when_engine_skill_is_missing(self):
        skills_root = self._install_skills(include_engine=False)
        script_path = skills_root / "chemqa-review" / "scripts" / "check_runtime.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--skill-root",
                str(skills_root / "chemqa-review"),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, result.returncode)
        payload = json.loads(result.stdout)
        self.assertIn("debateclaw-v1", payload["missing_skills"])
        self.assertFalse(payload["ready"])

    def test_compile_runplan_works_from_relocated_skills_root(self):
        skills_root = self._install_skills()
        script_path = skills_root / "chemqa-review" / "scripts" / "compile_runplan.py"
        payload = _run_json(
            [
                sys.executable,
                str(script_path),
                "--root",
                str(skills_root / "chemqa-review"),
                "--preset",
                "chemqa-review@1",
                "--goal",
                "Question: does Pt/C improve HER activity in 1 M KOH?",
                "--json",
            ]
        )
        self.assertEqual("chemqa-review@1", payload["workflow_ref"])
        self.assertEqual("review-loop@1", payload["engine_workflow_ref"])
        chemqa_context = payload["runtime_context"]["chemqa_review"]
        self.assertEqual("debateclaw-v1", chemqa_context["required_skills"][0])
        self.assertEqual("search_coverage", chemqa_context["role_map"]["proposer-2"])
        self.assertTrue(Path(chemqa_context["engine_skill_root"]).exists())

    def test_materialize_runplan_dry_run_generates_prompt_bundle_and_command_map(self):
        skills_root = self._install_skills()
        chemqa_root = skills_root / "chemqa-review"
        compile_script = chemqa_root / "scripts" / "compile_runplan.py"
        compile_payload = _run_json(
            [
                sys.executable,
                str(compile_script),
                "--root",
                str(chemqa_root),
                "--preset",
                "chemqa-review@1",
                "--goal",
                "Question: does Pt/C improve HER activity in 1 M KOH?",
                "--run-id",
                "chemqa-review-smoke",
                "--json",
            ]
        )
        self.assertEqual("chemqa-review-smoke", compile_payload["run_id"])

        fake_runtime_dir = skills_root / "runtime-bin"
        fake_runtime_dir.mkdir(parents=True, exist_ok=True)
        for helper_name in ("openclaw_debate_agent.py", "debate_state.py"):
            (fake_runtime_dir / helper_name).write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        materialize_script = chemqa_root / "scripts" / "materialize_runplan.py"
        payload = _run_json(
            [
                sys.executable,
                str(materialize_script),
                "--root",
                str(chemqa_root),
                "--run-id",
                "chemqa-review-smoke",
                "--runtime-dir",
                str(fake_runtime_dir),
                "--dry-run",
            ]
        )
        self.assertEqual("review-loop", payload["workflow_name"])
        self.assertTrue(Path(payload["command_map_path"]).exists())
        self.assertTrue(Path(payload["prompt_bundle_path"]).exists())
        prompt_bundle = json.loads(Path(payload["prompt_bundle_path"]).read_text(encoding="utf-8"))
        self.assertIn("debate-coordinator", prompt_bundle)
        self.assertIn("proposer-5", prompt_bundle)
        self.assertIn("reasoning_consistency", prompt_bundle["proposer-4"])

    def test_launch_from_preset_print_emits_clawteam_launch_command(self):
        skills_root = self._install_skills()
        chemqa_root = skills_root / "chemqa-review"
        fake_runtime_dir = skills_root / "runtime-bin"
        fake_runtime_dir.mkdir(parents=True, exist_ok=True)
        for helper_name in ("openclaw_debate_agent.py", "debate_state.py"):
            (fake_runtime_dir / helper_name).write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        script_path = chemqa_root / "scripts" / "launch_from_preset.py"
        env = dict(os.environ)
        env["HOME"] = str(skills_root / "home")
        payload = _run_json_with_env(
            [
                sys.executable,
                str(script_path),
                "--root",
                str(chemqa_root),
                "--preset",
                "chemqa-review@1",
                "--goal",
                "Question: does Pt/C improve HER activity in 1 M KOH?",
                "--run-id",
                "chemqa-review-launch",
                "--runtime-dir",
                str(fake_runtime_dir),
                "--launch-mode",
                "print",
            ],
            env=env,
        )
        self.assertEqual("chemqa-review-launch", payload["run_id"])
        self.assertEqual("clawteam", payload["launch_command"][0])
        self.assertEqual("launch", payload["launch_command"][1])
        self.assertIn("template_name", payload["materialize"])
        self.assertEqual(str(Path(env["HOME"]) / ".clawteam" / "templates"), payload["materialize"]["template_dir"])
        self.assertEqual(str(chemqa_root / "generated" / "clawteam-data"), payload["materialize"]["clawteam_data_dir"])

    def test_launch_from_preset_run_propagates_clawteam_data_dir(self):
        skills_root = self._install_skills()
        chemqa_root = skills_root / "chemqa-review"
        fake_runtime_dir = skills_root / "runtime-bin"
        fake_runtime_dir.mkdir(parents=True, exist_ok=True)
        for helper_name in ("openclaw_debate_agent.py", "debate_state.py"):
            (fake_runtime_dir / helper_name).write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        fake_bin_dir = skills_root / "fake-bin"
        fake_bin_dir.mkdir(parents=True, exist_ok=True)
        capture_path = skills_root / "clawteam-env.json"
        clawteam_script = fake_bin_dir / "clawteam"
        clawteam_script.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json",
                    "import os",
                    "import sys",
                    f"path = {str(capture_path)!r}",
                    "payload = {",
                    "    'argv': sys.argv,",
                    "    'clawteam_data_dir': os.environ.get('CLAWTEAM_DATA_DIR'),",
                    "}",
                    "with open(path, 'w', encoding='utf-8') as fh:",
                    "    json.dump(payload, fh)",
                    "print('fake clawteam launch ok')",
                    "raise SystemExit(0)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        clawteam_script.chmod(0o755)

        env = dict(os.environ)
        env["PATH"] = str(fake_bin_dir) + os.pathsep + env.get("PATH", "")

        script_path = chemqa_root / "scripts" / "launch_from_preset.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--root",
                str(chemqa_root),
                "--preset",
                "chemqa-review@1",
                "--goal",
                "Question: does Pt/C improve HER activity in 1 M KOH?",
                "--run-id",
                "chemqa-review-run-env",
                "--runtime-dir",
                str(fake_runtime_dir),
                "--template-dir",
                str(skills_root / "templates"),
                "--launch-mode",
                "run",
                "--json",
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(f"launch_from_preset failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        payload = json.loads(result.stdout)
        captured = json.loads(capture_path.read_text(encoding="utf-8"))
        expected_data_dir = str(chemqa_root / "generated" / "clawteam-data")
        self.assertEqual(expected_data_dir, payload["materialize"]["clawteam_data_dir"])
        self.assertEqual(expected_data_dir, captured["clawteam_data_dir"])

    def test_collect_artifacts_rebuilds_react_reviewed_protocol_files(self):
        skills_root = self._install_skills(include_engine=False)
        chemqa_root = skills_root / "chemqa-review"
        source_dir = skills_root / "run-source"
        output_dir = skills_root / "run-output"
        source_dir.mkdir(parents=True, exist_ok=True)

        protocol_payload = {
            "question": "Does Pt/C improve HER activity in 1 M KOH?",
            "final_answer": "Yes, Pt/C generally improves HER activity in 1 M KOH.",
            "acceptance_status": "accepted",
            "review_completion_status": "completed",
            "retrieval_diagnostics_summary": "",
            "execution_warnings": [],
            "sections": [],
            "citations": [],
            "claim_trace": [],
            "submission_trace": [],
            "overall_confidence": {
                "level": "medium",
                "score": 0.6,
                "rationale": "Fixture confidence.",
            },
            "section_confidence": [],
            "insufficient_evidence": False,
            "limitations_summary": "",
            "candidate_submission": {"submission_id": "submission_cycle_1"},
            "acceptance_decision": {"status": "accepted", "blocker_codes": [], "blocker_messages": [], "blocking_review_ids": []},
            "submission_cycles": [],
            "proposer_trajectory": {"trajectory_id": "traj-proposer", "query": "Does Pt/C improve HER activity in 1 M KOH?", "steps": []},
            "reviewer_trajectories": {},
            "review_statuses": [],
            "final_review_items": [],
        }
        (source_dir / "chemqa_review_protocol.json").write_text(
            json.dumps(protocol_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        script_path = chemqa_root / "scripts" / "collect_artifacts.py"
        payload = _run_json(
            [
                sys.executable,
                str(script_path),
                "--skill-root",
                str(chemqa_root),
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        self.assertTrue((output_dir / "qa_result.json").exists())
        self.assertTrue((output_dir / "candidate_submission.json").exists())
        qa_result = json.loads((output_dir / "qa_result.json").read_text(encoding="utf-8"))
        submission_trace = json.loads((output_dir / "submission_trace.json").read_text(encoding="utf-8"))
        review_statuses = json.loads((output_dir / "review_statuses.json").read_text(encoding="utf-8"))
        self.assertEqual("react_reviewed", qa_result["workflow_mode"])
        self.assertEqual("accepted", qa_result["acceptance_status"])
        self.assertEqual([], submission_trace)
        self.assertEqual([], review_statuses)
        self.assertEqual(str(output_dir / "qa_result.json"), payload["artifact_paths"]["qa_result"])


if __name__ == "__main__":
    unittest.main()
