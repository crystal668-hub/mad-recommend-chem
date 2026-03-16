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
        task_spec = self.router.run(question=question, context=context)
        entity_pack = self.entity_resolver.run(question=question, task_spec=task_spec)
        return GroundingState(
            question=question,
            context=context,
            task_spec=task_spec,
            entity_pack=entity_pack,
        )

    __call__ = run
