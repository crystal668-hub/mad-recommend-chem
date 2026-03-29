from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

from qa.runtime import DEFAULT_QA_MODEL_ALIASES
from qa.state import EntityPack, TaskSpec
from utils import Logger
import utils.logger as logger_mod


def confidence_payload(score: float = 0.82) -> dict[str, Any]:
    return {
        "level": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
        "score": score,
        "rationale": "test fixture",
    }


def make_task_spec(
    question: str = "How does Pt/C affect HER activity in 1 M KOH?",
    *,
    question_type: str = "fact",
) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "question": question,
            "normalized_question": question.lower(),
            "question_type": question_type,
            "recency_policy": "none",
            "answer_sections": [
                {
                    "section_id": "direct_answer",
                    "title": "Direct Answer",
                    "required": True,
                    "instruction": "Answer directly with the accepted evidence.",
                }
            ],
            "router_confidence": 0.92,
        }
    )


def make_entity_pack() -> EntityPack:
    return EntityPack.model_validate(
        {
            "entities": [
                {
                    "entity_id": "ent-1",
                    "mention": "Pt/C",
                    "canonical_name": "Pt/C",
                    "entity_type": "catalyst",
                    "aliases": ["platinum on carbon"],
                    "query_anchors": ["Pt/C", "platinum on carbon"],
                    "resolver_source": "seed",
                    "resolution_confidence": 0.96,
                    "status": "resolved",
                    "source_text": "Pt/C",
                    "source_span": {"start": 0, "end": 4},
                }
            ]
        }
    )


def make_base_config(root: Path, *, save_output: bool) -> Dict[str, Any]:
    outputs_dir = root / "outputs"
    return {
        "paths": {"outputs": str(outputs_dir)},
        "logging": {
            "level": "INFO",
            "log_file": str(root / "logs" / "system.log"),
            "run_dir": str(root / "logs" / "runs"),
        },
        "qa": {
            "workflow_mode": "react_reviewed",
            "save_output": save_output,
            "outputs_dir": str(outputs_dir),
            "artifact_subdir": "qa_artifacts",
            "models": copy.deepcopy(DEFAULT_QA_MODEL_ALIASES),
        },
        "llm": {},
    }


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_paths_exist(paths: Iterable[str | Path]) -> None:
    for path in paths:
        assert Path(path).exists(), str(path)


def reset_logging_state() -> None:
    logging.shutdown()
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    logger_mod._CONFIGURED = False
    logger_mod._RUN_DIR = None
    logger_mod._RUN_ID.set("")
    Logger._loggers.clear()


def flush_logging_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
