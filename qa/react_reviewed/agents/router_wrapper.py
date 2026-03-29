from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from qa.react_reviewed.common import QAArtifactStore, RouterExecutionError, RouterNode, TaskSpec

@dataclass
class RouterAgentWrapper:
    router: RouterNode

    def run(
        self,
        *,
        question: str,
        context: Optional[str],
        artifact_store: QAArtifactStore,
    ) -> Tuple[TaskSpec, Dict[str, str]]:
        try:
            task_spec = self.router.run(question=question, context=context)
        except RouterExecutionError as exc:
            debug_payload = dict(exc.debug_payload or {})
            semantic_stage_path = None
            localization_stage_path = None
            fallback_reason_path = None
            if isinstance(debug_payload.get("semantic_stage"), dict):
                semantic_stage_path = artifact_store.write_json(
                    "router/semantic_stage.json",
                    debug_payload["semantic_stage"],
                )
            if isinstance(debug_payload.get("localization_stage"), dict):
                localization_stage_path = artifact_store.write_json(
                    "router/localization_stage.json",
                    debug_payload["localization_stage"],
                )
            if isinstance(debug_payload.get("fallback_reason"), dict):
                fallback_reason_path = artifact_store.write_json(
                    "router/fallback_reason.json",
                    debug_payload["fallback_reason"],
                )
            failure_path = artifact_store.write_json(
                "router/failure.json",
                exc.to_payload(),
            )
            artifact_store.write_json(
                "router/agent_run.json",
                {
                    "agent": "RouterAgent",
                    "input": {"question": question, "context": context},
                    "error": exc.to_payload(),
                    "debug": debug_payload,
                },
            )
            if semantic_stage_path:
                debug_payload["semantic_stage_artifact"] = semantic_stage_path
            if localization_stage_path:
                debug_payload["localization_stage_artifact"] = localization_stage_path
            if fallback_reason_path:
                debug_payload["fallback_reason_artifact"] = fallback_reason_path
            debug_payload["failure_artifact"] = failure_path
            raise

        debug_payload = dict(getattr(self.router, "last_run_debug", {}) or {})
        task_spec_path = artifact_store.write_json(
            "router/task_spec.json",
            task_spec.model_dump(exclude_none=True),
        )
        semantic_stage_path = None
        localization_stage_path = None
        fallback_reason_path = None
        if isinstance(debug_payload.get("semantic_stage"), dict):
            semantic_stage_path = artifact_store.write_json(
                "router/semantic_stage.json",
                debug_payload["semantic_stage"],
            )
        if isinstance(debug_payload.get("localization_stage"), dict):
            localization_stage_path = artifact_store.write_json(
                "router/localization_stage.json",
                debug_payload["localization_stage"],
            )
        if isinstance(debug_payload.get("fallback_reason"), dict):
            fallback_reason_path = artifact_store.write_json(
                "router/fallback_reason.json",
                debug_payload["fallback_reason"],
            )
        audit_path = artifact_store.write_json(
            "router/agent_run.json",
            {
                "agent": "RouterAgent",
                "input": {"question": question, "context": context},
                "output": task_spec.model_dump(exclude_none=True),
                "debug": debug_payload,
            },
        )
        artifacts = {
            "router_task_spec": task_spec_path,
            "router_agent_run": audit_path,
        }
        if semantic_stage_path:
            artifacts["router_semantic_stage"] = semantic_stage_path
        if localization_stage_path:
            artifacts["router_localization_stage"] = localization_stage_path
        if fallback_reason_path:
            artifacts["router_fallback_reason"] = fallback_reason_path
        return task_spec, artifacts
