from __future__ import annotations

from typing import Optional

from qa.nodes.entity_resolver import EntityResolverNode
from qa.nodes.router import RouterNode
from qa.state import GroundingState


class QueryGroundingPipeline:
    def __init__(
        self,
        router: Optional[RouterNode] = None,
        entity_resolver: Optional[EntityResolverNode] = None,
    ) -> None:
        self.router = router or RouterNode()
        self.entity_resolver = entity_resolver or EntityResolverNode()

    def run(self, question: str, context: Optional[str] = None) -> GroundingState:
        grounding_state, _ = self.run_detailed(question=question, context=context)
        return grounding_state

    def run_detailed(self, question: str, context: Optional[str] = None):
        task_spec = self.router.run(question=question, context=context)
        resolve_detailed = getattr(self.entity_resolver, "resolve_detailed", None)
        if callable(resolve_detailed):
            resolution_result = resolve_detailed(question=question, task_spec=task_spec)
            entity_pack = resolution_result.entity_pack
        else:
            entity_pack = self.entity_resolver.run(question=question, task_spec=task_spec)
            resolution_result = None
        grounding_state = GroundingState(
            question=question,
            context=context,
            task_spec=task_spec,
            entity_pack=entity_pack,
        )
        return grounding_state, resolution_result

    __call__ = run
