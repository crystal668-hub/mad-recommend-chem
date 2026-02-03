import unittest
from collections import Counter

from debate.langgraph_coordinator import LangGraphDebateCoordinator


class LangGraphReviewAssignmentTests(unittest.TestCase):
    def _coordinator(self) -> LangGraphDebateCoordinator:
        # No agents needed: we only test deterministic assignment logic.
        return LangGraphDebateCoordinator(agents=[], config={})

    def test_assignments_cover_all_active_proposals(self):
        coord = self._coordinator()
        active_ids = ["agent1", "agent2", "agent3", "agent4"]

        assignments = coord._assign_review_targets(round_number=1, active_ids=active_ids)

        # Every active proposal is a reviewer.
        self.assertEqual(set(assignments.keys()), set(active_ids))

        # Every active proposal is also reviewed by someone (exactly once for the current strategy).
        assigned_targets = [t for targets in assignments.values() for t in targets]
        self.assertEqual(set(assigned_targets), set(active_ids))
        self.assertTrue(all(v == 1 for v in Counter(assigned_targets).values()))

        # No self-review.
        for reviewer_id, targets in assignments.items():
            self.assertNotIn(reviewer_id, targets)

    def test_rotation_changes_pairings_across_rounds(self):
        coord = self._coordinator()
        active_ids = ["agent1", "agent2", "agent3", "agent4"]

        a1 = coord._assign_review_targets(round_number=1, active_ids=active_ids)
        a2 = coord._assign_review_targets(round_number=2, active_ids=active_ids)
        a3 = coord._assign_review_targets(round_number=3, active_ids=active_ids)

        self.assertNotEqual(a1, a2)
        self.assertNotEqual(a2, a3)

    def test_single_active_returns_empty(self):
        coord = self._coordinator()
        self.assertEqual(coord._assign_review_targets(round_number=1, active_ids=["agent1"]), {})


if __name__ == "__main__":
    unittest.main()

