from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from qa.react_reviewed.common import EntityPack, EntityResolverNode, QAArtifactStore, TaskSpec

@dataclass
class EntityResolverAgentWrapper:
    resolver: EntityResolverNode

    def run(
        self,
        *,
        question: str,
        task_spec: TaskSpec,
        artifact_store: QAArtifactStore,
    ) -> Tuple[EntityPack, Dict[str, str], Dict[str, Any]]:
        resolve_detailed = getattr(self.resolver, "resolve_detailed", None)
        if callable(resolve_detailed):
            resolution_result = resolve_detailed(question=question, task_spec=task_spec)
            entity_pack = resolution_result.entity_pack
            artifact_payloads = dict(resolution_result.artifact_payloads())
            resolution_snapshot = {
                "resolution_index": artifact_payloads.get("entity_resolver/resolution_index.json", {}),
                "provider_calls": artifact_payloads.get("entity_resolver/provider_calls.json", []),
                "seed_suggestions": artifact_payloads.get("entity_resolver/seed_suggestions.json", []),
            }
        else:
            entity_pack = self.resolver.run(question=question, task_spec=task_spec)
            artifact_payloads = {
                "entity_resolver/entity_pack.json": entity_pack.model_dump(exclude_none=True),
                "entity_resolver/resolution_index.json": {"entries": [], "cache_events": []},
                "entity_resolver/provider_calls.json": [],
                "entity_resolver/seed_suggestions.json": [],
            }
            resolution_snapshot = {
                "resolution_index": artifact_payloads["entity_resolver/resolution_index.json"],
                "provider_calls": [],
                "seed_suggestions": [],
            }
        artifact_paths = {
            Path(relative_path).stem: artifact_store.write_json(relative_path, payload)
            for relative_path, payload in artifact_payloads.items()
        }
        audit_path = artifact_store.write_json(
            "entity_resolver/agent_run.json",
            {
                "agent": "EntityResolverAgent",
                "input": {
                    "question": question,
                    "task_spec": task_spec.model_dump(exclude_none=True),
                },
                "output": entity_pack.model_dump(exclude_none=True),
            },
        )
        artifact_paths["entity_resolver_agent_run"] = audit_path
        return entity_pack, artifact_paths, resolution_snapshot
