import unittest

from hotel_cancellation.decision import (
    expected_intervention_value,
    recommendation,
    risk_band,
    threshold_table,
)


class DecisionLogicTests(unittest.TestCase):
    def test_risk_bands_cover_probability_range(self):
        self.assertEqual(risk_band(0.10), "Low")
        self.assertEqual(risk_band(0.30), "Medium")
        self.assertEqual(risk_band(0.50), "High")
        self.assertEqual(risk_band(0.70), "Very high")

    def test_expected_value_and_recommendation(self):
        value = expected_intervention_value(0.8, 100.0, 5.0, 0.25)
        self.assertAlmostEqual(value, 15.0)
        self.assertEqual(
            recommendation(0.8, 0.5, value),
            "Prioritize for retention review",
        )

    def test_threshold_table_counts_and_value(self):
        result = threshold_table(
            y_true=[1, 1, 0, 0],
            probabilities=[0.9, 0.6, 0.7, 0.2],
            recoverable_margin=100.0,
            intervention_cost=5.0,
            intervention_success_rate=0.5,
            thresholds=[0.5],
        ).iloc[0]
        self.assertEqual(result["interventions"], 3)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["missed_cancellations"], 0)
        self.assertAlmostEqual(result["assumed_net_value"], 85.0)


if __name__ == "__main__":
    unittest.main()
