from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from qa.artifacts import QAArtifactStore
from qa.runtime import QARuntime, build_qa_runtime, resolve_qa_runtime_config
from qa.synthesis_state import QAResult
from utils import ensure_dir, generate_timestamp, get_run_dir, load_config, save_json


logger = logging.getLogger("MAD.qa.system")


class QASystem:
    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "./config/config.yaml",
        grounding_pipeline: Optional[Any] = None,
        query_planner: Optional[Any] = None,
        retriever: Optional[Any] = None,
        document_acquirer: Optional[Any] = None,
        handoff: Optional[Any] = None,
        evidence_extractor: Optional[Any] = None,
        react_reviewed_workflow: Optional[Any] = None,
    ) -> None:
        loaded_config = copy.deepcopy(config) if config is not None else load_config(config_path)
        self.config = loaded_config
        self.config_path = config_path
        self.qa_config = resolve_qa_runtime_config(loaded_config)

        runtime = build_qa_runtime(
            config=self.config,
            config_path=self.config_path,
            grounding_pipeline=grounding_pipeline,
            query_planner=query_planner,
            retriever=retriever,
            document_acquirer=document_acquirer,
            handoff=handoff,
            evidence_extractor=evidence_extractor,
            react_reviewed_workflow=react_reviewed_workflow,
        )
        self.runtime: QARuntime = runtime
        self.runtime_manifest = copy.deepcopy(runtime.runtime_manifest)
        self.grounding_pipeline = runtime.grounding_pipeline
        self.query_planner = runtime.query_planner
        self.retriever = runtime.retriever
        self.document_acquirer = runtime.document_acquirer
        self.handoff = runtime.handoff
        self.evidence_extractor = runtime.evidence_extractor
        self.react_reviewed_workflow = runtime.react_reviewed_workflow

    def run_qa(
        self,
        question: str,
        context: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> QAResult:
        resolved_artifact_dir = self._resolve_artifact_dir(artifact_dir)
        runtime_manifest_path = self._write_runtime_manifest(resolved_artifact_dir)
        logger.info("qa_run_start artifact_dir=%s", resolved_artifact_dir)

        result = self.react_reviewed_workflow.run(
            question=question,
            context=context,
            artifact_dir=resolved_artifact_dir,
        )
        artifact_paths = dict(result.artifact_paths)
        artifact_paths["runtime_manifest"] = runtime_manifest_path
        execution_warnings = self._merge_execution_warnings(
            result.execution_warnings,
            self.runtime_manifest.get("warnings"),
        )

        if self.qa_config["save_output"]:
            public_result_path = self._public_result_path()
            artifact_paths["public_result"] = str(public_result_path)

        finalized_result = result.model_copy(
            update={
                "artifact_paths": artifact_paths,
                "execution_warnings": execution_warnings,
            }
        )

        if self.qa_config["save_output"]:
            self._write_public_result(
                result=finalized_result,
                artifact_paths=artifact_paths,
                destination=public_result_path,
            )

        qa_result_path = artifact_paths.get("qa_result")
        if qa_result_path:
            save_json(finalized_result.model_dump(exclude_none=True), qa_result_path)

        logger.info(
            "qa_run_complete artifact_dir=%s warnings=%s",
            resolved_artifact_dir,
            len(finalized_result.execution_warnings),
        )
        return finalized_result

    __call__ = run_qa

    def _resolve_artifact_dir(self, artifact_dir: Optional[str]) -> str:
        if artifact_dir:
            return str(Path(artifact_dir))

        run_dir = get_run_dir()
        artifact_subdir = self.qa_config["artifact_subdir"]
        if run_dir is not None:
            return str(run_dir / artifact_subdir)
        return str(Path("./logs/runs") / generate_timestamp() / artifact_subdir)

    def _write_runtime_manifest(self, artifact_dir: str) -> str:
        store = QAArtifactStore(base_dir=artifact_dir)
        return store.write_json("runtime_manifest.json", self.runtime_manifest)

    def _public_result_path(self) -> Path:
        output_dir = ensure_dir(self.qa_config["outputs_dir"])
        return output_dir / f"qa_result_{generate_timestamp()}.json"

    def _write_public_result(
        self,
        *,
        result: QAResult,
        artifact_paths: Dict[str, str],
        destination: Path,
    ) -> Path:
        payload = result.model_copy(update={"artifact_paths": artifact_paths}).model_dump(exclude_none=True)
        save_json(payload, destination)
        return destination

    def _merge_execution_warnings(
        self,
        existing_warnings: Optional[list[str]],
        runtime_warnings: Optional[list[str]],
    ) -> list[str]:
        merged: list[str] = []
        seen = set()
        for warning in [*(existing_warnings or []), *(runtime_warnings or [])]:
            text = str(warning or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged


def run_qa(
    question: str,
    context: Optional[str] = None,
    artifact_dir: Optional[str] = None,
) -> QAResult:
    system = QASystem()
    return system.run_qa(question=question, context=context, artifact_dir=artifact_dir)
