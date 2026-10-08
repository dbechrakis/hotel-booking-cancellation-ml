import unittest

import numpy as np

from hotel_cancellation.calibration import (
    calibration_summary,
    expected_calibration_error,
    reliability_table,
)
from hotel_cancellation.model import fit_calibrator


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.true_probability = rng.uniform(0, 1, 20000)
        self.outcomes = (rng.uniform(0, 1, 20000) < self.true_probability).astype(int)

    def test_true_probabilities_have_small_error(self):
        self.assertLess(expected_calibration_error(self.outcomes, self.true_probability), 0.02)

    def test_inflated_scores_are_detected_and_repaired(self):
        inflated = np.sqrt(self.true_probability)  # monotone, overconfident in cancellation
        self.assertGreater(expected_calibration_error(self.outcomes, inflated), 0.1)
        half = len(inflated) // 2
        calibrator = fit_calibrator(inflated[:half], self.outcomes[:half])
        repaired = calibrator.predict(inflated[half:])
        self.assertLess(expected_calibration_error(self.outcomes[half:], repaired), 0.03)
        self.assertLess(
            calibration_summary(self.outcomes[half:], repaired)["brier"],
            calibration_summary(self.outcomes[half:], inflated[half:])["brier"],
        )

    def test_reliability_bins_ignore_row_order_with_ties(self):
        steps = np.round(self.true_probability, 1)
        order = np.random.default_rng(1).permutation(len(steps))
        a = reliability_table(self.outcomes, steps)
        b = reliability_table(self.outcomes[order], steps[order])
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())

    def test_calibrator_clips_out_of_range_scores(self):
        calibrator = fit_calibrator(np.array([0.2, 0.4, 0.6, 0.8]), np.array([0, 0, 1, 1]))
        self.assertTrue(np.all((calibrator.predict(np.array([-1.0, 2.0])) >= 0) & (calibrator.predict(np.array([-1.0, 2.0])) <= 1)))


if __name__ == "__main__":
    unittest.main()
