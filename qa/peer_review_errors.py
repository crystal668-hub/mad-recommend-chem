from __future__ import annotations

from typing import Any, Dict, Optional


class PeerReviewExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        run_stats: Optional[Dict[str, Any]] = None,
        reviewer_type: Optional[str] = None,
        claim_id: Optional[str] = None,
        review_round: Optional[int] = None,
        response_content: Any = None,
    ) -> None:
        super().__init__(str(message))
        self.stage = str(stage)
        self.details = dict(details or {})
        self.run_stats = dict(run_stats or {})
        self.reviewer_type = str(reviewer_type or "").strip() or None
        self.claim_id = str(claim_id or "").strip() or None
        self.review_round = int(review_round) if review_round is not None else None
        self.response_content = response_content

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": "peer_review_execution_failed",
            "stage": self.stage,
            "reason": str(self),
            "details": dict(self.details),
            "run_stats": dict(self.run_stats),
            "reviewer_type": self.reviewer_type,
            "claim_id": self.claim_id,
            "review_round": self.review_round,
            "response_content": self.response_content,
        }
