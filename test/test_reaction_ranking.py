import unittest


class ReactionRankingTests(unittest.TestCase):
    def test_grade_precedence(self):
        from utils.reaction_ranking import rank_reactions

        items = [
            {
                "reaction_type": "OER",
                "performance_evaluation": {"reaction_type": "OER", "grade": "Good", "metric_value": 240.0, "metric_unit": "mV"},
            },
            {
                "reaction_type": "HER",
                "performance_evaluation": {"reaction_type": "HER", "grade": "Outstanding", "metric_value": 60.0, "metric_unit": "mV"},
            },
        ]
        ranking, top = rank_reactions(items, top_k=2)
        self.assertEqual(ranking[0].get("reaction_type"), "HER")
        self.assertEqual(top[0].get("reaction_type"), "HER")

    def test_tie_break_lower_is_better(self):
        from utils.reaction_ranking import rank_reactions

        items = [
            {
                "reaction_type": "OER",
                "performance_evaluation": {"reaction_type": "OER", "grade": "Good", "metric_value": 240.0, "metric_unit": "mV"},
            },
            {
                "reaction_type": "HER",
                "performance_evaluation": {"reaction_type": "HER", "grade": "Good", "metric_value": 90.0, "metric_unit": "mV"},
            },
        ]
        ranking, _top = rank_reactions(items, top_k=2)
        # Both are lower-is-better metrics; smaller metric_value should rank higher.
        self.assertEqual(ranking[0].get("reaction_type"), "HER")

    def test_missing_metric_sorts_last_within_grade(self):
        from utils.reaction_ranking import rank_reactions

        items = [
            {
                "reaction_type": "OER",
                "performance_evaluation": {"reaction_type": "OER", "grade": "Good", "metric_value": 240.0, "metric_unit": "mV"},
            },
            {
                "reaction_type": "HER",
                "performance_evaluation": {"reaction_type": "HER", "grade": "Good"},
            },
        ]
        ranking, _top = rank_reactions(items, top_k=2)
        self.assertEqual(ranking[-1].get("reaction_type"), "HER")


if __name__ == "__main__":
    unittest.main()

