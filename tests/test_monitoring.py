import unittest

import numpy as np

from hotel_cancellation.monitoring import (
    ScoreMonitor,
    drift_status,
    population_stability_index,
    reference_profile,
)


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.reference_scores = rng.beta(2, 3, 5000)
        self.reference = reference_profile(self.reference_scores, self.reference_scores * 0.8, threshold=0.4)

    def test_reference_bins_are_deciles(self):
        self.assertEqual(len(self.reference["bin_edges"]), 9)
        np.testing.assert_allclose(self.reference["bin_shares"], 0.1, atol=0.002)

    def test_same_distribution_is_stable_and_shift_alerts(self):
        rng = np.random.default_rng(1)
        same = population_stability_index(self.reference["bin_shares"], rng.beta(2, 3, 5000), self.reference["bin_edges"])
        shifted = population_stability_index(self.reference["bin_shares"], rng.beta(4, 2, 5000), self.reference["bin_edges"])
        self.assertEqual(drift_status(same), "ok")
        self.assertEqual(drift_status(shifted), "alert")

    def test_monitor_needs_enough_rows_before_judging(self):
        monitor = ScoreMonitor(self.reference, window=1000, min_rows=200)
        self.assertEqual(monitor.summary()["status"], "no_traffic")
        monitor.record(np.full(50, 0.3), np.full(50, 0.2), np.zeros(50, bool), np.zeros(50, bool), 3.0)
        self.assertEqual(monitor.summary()["status"], "insufficient_data")

    def test_monitor_window_summary(self):
        monitor = ScoreMonitor(self.reference, window=1000, min_rows=200)
        scores = np.random.default_rng(2).beta(2, 3, 1500)
        monitor.record(scores, scores * 0.8, scores >= 0.4, scores > 0.95, 4.0)
        summary = monitor.summary()
        self.assertEqual(summary["window_rows"], 1000)  # rolling window keeps the latest scores
        self.assertEqual(summary["bookings_scored_total"], 1500)
        self.assertAlmostEqual(summary["flagged_share"], float(np.mean(scores[-1000:] >= 0.4)))
        self.assertEqual(summary["status"], "ok")


if __name__ == "__main__":
    unittest.main()
