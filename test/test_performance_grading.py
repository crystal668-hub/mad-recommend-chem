import unittest


class PerformanceGradingTests(unittest.TestCase):
    def test_grade_boundaries(self):
        from utils.performance_grading import grade_value

        # HER (lower is better)
        self.assertEqual(grade_value("HER", 50), "Outstanding")
        self.assertEqual(grade_value("HER", 100), "Good")
        self.assertEqual(grade_value("HER", 200), "Fair")
        self.assertEqual(grade_value("HER", 250), "Poor")
        self.assertEqual(grade_value("HER", 251), "Terrible")

        # OER (lower is better)
        self.assertEqual(grade_value("OER", 200), "Outstanding")
        self.assertEqual(grade_value("OER", 250), "Good")
        self.assertEqual(grade_value("OER", 300), "Fair")
        self.assertEqual(grade_value("OER", 350), "Poor")
        self.assertEqual(grade_value("OER", 351), "Terrible")

        # ORR (higher is better)
        self.assertEqual(grade_value("ORR", 0.92), "Outstanding")
        self.assertEqual(grade_value("ORR", 0.85), "Good")
        self.assertEqual(grade_value("ORR", 0.75), "Fair")
        self.assertEqual(grade_value("ORR", 0.65), "Poor")
        self.assertEqual(grade_value("ORR", 0.649), "Terrible")

        # HZOR (more negative is better)
        self.assertEqual(grade_value("HZOR", -100), "Outstanding")
        self.assertEqual(grade_value("HZOR", 0), "Good")
        self.assertEqual(grade_value("HZOR", 50), "Fair")
        self.assertEqual(grade_value("HZOR", 100), "Poor")
        self.assertEqual(grade_value("HZOR", 101), "Terrible")

    def test_parse_overpotential_mv(self):
        from utils.performance_grading import parse_metric_value

        v, u = parse_metric_value("OER", "310 mV overpotential at 10 mA/cm^2")
        self.assertAlmostEqual(v, 310.0)
        self.assertEqual(u, "mV")

    def test_parse_overpotential_plus_minus_prefers_point_estimate(self):
        from utils.performance_grading import parse_metric_value

        raw = (
            "overpotential 291.5 \u00b1 0.5 mV at 10 mA cm\u22122; "
            "Tafel slope 43.9 mV dec\u22121"
        )
        v, u = parse_metric_value("OER", raw)
        self.assertAlmostEqual(v, 291.5)
        self.assertEqual(u, "mV")

    def test_parse_orr_e_half(self):
        from utils.performance_grading import parse_metric_value

        v, u = parse_metric_value("ORR", "E1/2 = 0.89 V (vs RHE)")
        self.assertAlmostEqual(v, 0.89)
        self.assertEqual(u, "V")

        v2, u2 = parse_metric_value("ORR", "half-wave potential: 890 mV")
        self.assertAlmostEqual(v2, 0.89)
        self.assertEqual(u2, "V")

    def test_parse_hor_exchange_current_density(self):
        from utils.performance_grading import parse_metric_value

        v, u = parse_metric_value("HOR", "j0 = 2.7 mA cm-2")
        self.assertAlmostEqual(v, 2.7)
        self.assertEqual(u, "mA cm-2")

        v2, u2 = parse_metric_value("HOR", "exchange current density (j0): 0.003 A/cm^2")
        self.assertAlmostEqual(v2, 3.0)
        self.assertEqual(u2, "mA cm-2")

    def test_parse_eor_mass_activity(self):
        from utils.performance_grading import parse_metric_value

        v, u = parse_metric_value("EOR", "Mass activity: 28 mA/mg_metal")
        self.assertAlmostEqual(v, 0.028)
        self.assertEqual(u, "A mgmetal-1")

    def test_parse_o5h_fe_ratio_to_percent(self):
        from utils.performance_grading import parse_metric_value

        v, u = parse_metric_value("O5H", "FE = 0.95")
        self.assertAlmostEqual(v, 95.0)
        self.assertEqual(u, "%")

        v2, u2 = parse_metric_value("O5H", "Faradaic efficiency: 92%")
        self.assertAlmostEqual(v2, 92.0)
        self.assertEqual(u2, "%")

    def test_extract_last_performance_metrics_wins(self):
        from utils.performance_grading import extract_last_performance_metrics_text

        claim = (
            "Reaction Type: OER\n"
            "Performance Metrics: 360 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
            "Performance Metrics: 350 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
        )
        raw = extract_last_performance_metrics_text(claim)
        self.assertIn("350", raw or "")

    def test_evaluate_claim_none_when_missing(self):
        from utils.performance_grading import evaluate_claim

        self.assertIsNone(evaluate_claim(""))
        self.assertIsNone(evaluate_claim("Reaction Type: OER\nProducts: O2\n"))

    def test_evaluate_claim_happy_path(self):
        from utils.performance_grading import evaluate_claim

        claim = (
            "Reaction Type: OER\n"
            "Products: O2\n"
            "Performance Metrics: 310 mV overpotential at 10 mA/cm^2 (Confidence: low)\n"
        )
        out = evaluate_claim(claim)
        self.assertIsInstance(out, dict)
        self.assertEqual(out.get("reaction_type"), "OER")
        self.assertAlmostEqual(out.get("metric_value"), 310.0)
        self.assertEqual(out.get("metric_unit"), "mV")
        self.assertEqual(out.get("grade"), "Poor")


if __name__ == "__main__":
    unittest.main()

