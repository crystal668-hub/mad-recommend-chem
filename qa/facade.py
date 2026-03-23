from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from qa.artifacts import QAArtifactStore
from qa.nodes.router import RouterExecutionError
from qa.peer_review_errors import PeerReviewExecutionError
from qa.retrieval_state import RetrievalState
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
        retrieval_pipeline: Optional[Any] = None,
        peer_review_pipeline: Optional[Any] = None,
        synthesis_pipeline: Optional[Any] = None,
        react_reviewed_workflow: Optional[Any] = None,
    ) -> None:
        loaded_config = copy.deepcopy(config) if config is not None else load_config(config_path)
        self.config = loaded_config
        self.config_path = config_path
        self.qa_config = resolve_qa_runtime_config(loaded_config)

        runtime = self._build_runtime(
            grounding_pipeline=grounding_pipeline,
            retrieval_pipeline=retrieval_pipeline,
            peer_review_pipeline=peer_review_pipeline,
            synthesis_pipeline=synthesis_pipeline,
            react_reviewed_workflow=react_reviewed_workflow,
        )
        self.runtime: QARuntime = runtime
        self.runtime_manifest = copy.deepcopy(runtime.runtime_manifest)
        self.grounding_pipeline = runtime.grounding_pipeline
        self.retrieval_pipeline = self._configure_retrieval_pipeline(runtime.retrieval_pipeline)
        self.peer_review_pipeline = runtime.peer_review_pipeline
        self.synthesis_pipeline = runtime.synthesis_pipeline
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
        if self.qa_config["workflow_mode"] == "react_reviewed":
            if self.react_reviewed_workflow is None:
                raise ValueError("react_reviewed workflow mode requires a react_reviewed_workflow runtime.")
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

        run_detailed = getattr(self.grounding_pipeline, "run_detailed", None)
        try:
            if callable(run_detailed):
                grounding_state, resolution_result = run_detailed(question=question, context=context)
            else:
                grounding_state = self.grounding_pipeline.run(question=question, context=context)
                resolution_result = None
        except RouterExecutionError as exc:
            self._write_router_failure_artifacts(
                artifact_dir=resolved_artifact_dir,
                question=question,
                context=context,
                error=exc,
            )
            raise
        entity_artifacts = self._write_entity_resolution_artifacts(
            artifact_dir=resolved_artifact_dir,
            resolution_result=resolution_result,
        )
        logger.info("qa_grounding_complete question_type=%s", grounding_state.task_spec.question_type)
        retrieval_state = self.retrieval_pipeline.run_from_grounding(
            grounding_state,
            artifact_dir=resolved_artifact_dir,
        )

        if retrieval_state.evidence_ledger is None:
            raise ValueError("Retrieval pipeline must return an evidence ledger before synthesis.")

        review_artifacts: Dict[str, str] = {}
        execution_warnings: list[str] = []
        if self.peer_review_pipeline is not None:
            logger.info("qa_peer_review_dispatch claims=%s", len(retrieval_state.evidence_ledger.claims))
            try:
                reviewed_ledger = self.peer_review_pipeline.run(
                    retrieval_state.evidence_ledger,
                    task_spec=grounding_state.task_spec,
                )
            except PeerReviewExecutionError as exc:
                self._write_peer_review_failure_artifacts(
                    artifact_dir=resolved_artifact_dir,
                    evidence_ledger=retrieval_state.evidence_ledger,
                    task_spec=grounding_state.task_spec,
                    error=exc,
                )
                raise
            retrieval_state = retrieval_state.model_copy(update={"evidence_ledger": reviewed_ledger})
            review_artifacts = self._write_review_artifacts(retrieval_state)
            execution_warnings = self._merge_execution_warnings(
                execution_warnings,
                getattr(self.peer_review_pipeline, "last_execution_warnings", None),
            )

        result = self.synthesis_pipeline.run_from_retrieval(
            retrieval_state,
            artifact_dir=retrieval_state.artifact_dir,
            execution_warnings=execution_warnings,
        )

        artifact_paths = dict(result.artifact_paths)
        artifact_paths.update(entity_artifacts)
        artifact_paths.update(review_artifacts)
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

    def _build_runtime(
        self,
        *,
        grounding_pipeline: Optional[Any],
        retrieval_pipeline: Optional[Any],
        peer_review_pipeline: Optional[Any],
        synthesis_pipeline: Optional[Any],
        react_reviewed_workflow: Optional[Any],
    ) -> QARuntime:
        return build_qa_runtime(
            config=self.config,
            config_path=self.config_path,
            grounding_pipeline=grounding_pipeline,
            retrieval_pipeline=retrieval_pipeline,
            peer_review_pipeline=peer_review_pipeline,
            synthesis_pipeline=synthesis_pipeline,
            react_reviewed_workflow=react_reviewed_workflow,
        )

    def _configure_retrieval_pipeline(
        self,
        retrieval_pipeline: Any,
    ) -> Any:
        pipeline = retrieval_pipeline
        if getattr(pipeline, "peer_review_pipeline", None) is not None:
            setattr(pipeline, "peer_review_pipeline", None)
        return pipeline

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

    def _write_review_artifacts(self, retrieval_state: RetrievalState) -> Dict[str, str]:
        store = QAArtifactStore(base_dir=retrieval_state.artifact_dir)
        evidence_ledger = retrieval_state.evidence_ledger
        if evidence_ledger is None:
            return {}

        reviewed_ledger_path = store.write_json(
            "evidence_ledger_reviewed.json",
            evidence_ledger.model_dump(exclude_none=True),
        )
        review_summaries_path = store.write_json(
            "review_summaries.json",
            [item.model_dump(exclude_none=True) for item in evidence_ledger.review_summaries],
        )
        return {
            "reviewed_evidence_ledger": reviewed_ledger_path,
            "review_summaries": review_summaries_path,
        }

    def _write_entity_resolution_artifacts(
        self,
        *,
        artifact_dir: str,
        resolution_result: Optional[Any],
    ) -> Dict[str, str]:
        if resolution_result is None or not hasattr(resolution_result, "artifact_payloads"):
            return {}
        store = QAArtifactStore(base_dir=artifact_dir)
        artifact_paths: Dict[str, str] = {}
        for relative_path, payload in dict(resolution_result.artifact_payloads()).items():
            artifact_paths[Path(relative_path).stem] = store.write_json(relative_path, payload)
        return artifact_paths

    def _write_router_failure_artifacts(
        self,
        *,
        artifact_dir: str,
        question: str,
        context: Optional[str],
        error: RouterExecutionError,
    ) -> Dict[str, str]:
        store = QAArtifactStore(base_dir=artifact_dir)
        debug_payload = dict(error.debug_payload or {})
        artifact_paths: Dict[str, str] = {}
        if isinstance(debug_payload.get("semantic_stage"), dict):
            artifact_paths["router_semantic_stage"] = store.write_json(
                "router/semantic_stage.json",
                debug_payload["semantic_stage"],
            )
        if isinstance(debug_payload.get("localization_stage"), dict):
            artifact_paths["router_localization_stage"] = store.write_json(
                "router/localization_stage.json",
                debug_payload["localization_stage"],
            )
        artifact_paths["router_failure"] = store.write_json(
            "router/failure.json",
            error.to_payload(),
        )
        artifact_paths["router_agent_run"] = store.write_json(
            "router/agent_run.json",
            {
                "agent": "RouterAgent",
                "input": {"question": question, "context": context},
                "error": error.to_payload(),
                "debug": debug_payload,
            },
        )
        return artifact_paths

    def _write_peer_review_failure_artifacts(
        self,
        *,
        artifact_dir: str,
        evidence_ledger: Any,
        task_spec: Any,
        error: PeerReviewExecutionError,
    ) -> Dict[str, str]:
        store = QAArtifactStore(base_dir=artifact_dir)
        ledger_payload = (
            evidence_ledger.model_dump(exclude_none=True)
            if hasattr(evidence_ledger, "model_dump")
            else dict(evidence_ledger or {})
        )
        task_spec_payload = (
            task_spec.model_dump(exclude_none=True)
            if task_spec is not None and hasattr(task_spec, "model_dump")
            else None
        )
        artifact_paths: Dict[str, str] = {}
        artifact_paths["peer_review_failure"] = store.write_json(
            "peer_review/failure.json",
            error.to_payload(),
        )
        artifact_paths["peer_review_agent_run"] = store.write_json(
            "peer_review/agent_run.json",
            {
                "agent": "StructuredPeerReviewPipeline",
                "input": {
                    "task_spec": task_spec_payload,
                    "claim_count": len(list(getattr(evidence_ledger, "claims", []) or [])),
                    "evidence_count": len(list(getattr(evidence_ledger, "evidence_items", []) or [])),
                    "evidence_ledger": ledger_payload,
                },
                "error": error.to_payload(),
            },
        )
        return artifact_paths

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
