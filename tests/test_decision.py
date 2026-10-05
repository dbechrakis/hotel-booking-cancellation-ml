import unittest

from hotel_cancellation.decision import (
    expected_intervention_value,
    experiment_sample_size,
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


    def test_experiment_sample_size_matches_textbook_value(self):
        # 50% -> 40%, two-sided alpha 0.05, power 0.80: 388 per arm.
        self.assertEqual(experiment_sample_size(0.5, 0.2), 388)

    def test_experiment_sample_size_grows_for_smaller_effects(self):
        sizes = [experiment_sample_size(0.47, r) for r in (0.30, 0.20, 0.10, 0.05)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(experiment_sample_size(0.47, 0.1, power=0.9), sizes[2])

    def test_experiment_sample_size_rejects_invalid_inputs(self):
        for args in ((0.0, 0.1), (1.0, 0.1), (0.5, 0.0), (0.5, 1.0)):
            with self.assertRaises(ValueError):
                experiment_sample_size(*args)
        with self.assertRaises(ValueError):
            experiment_sample_size(0.5, 0.1, alpha=0)


if __name__ == "__main__":
    unittest.main()
