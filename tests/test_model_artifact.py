from pathlib import Path
import os
import unittest

import numpy as np
import pandas as pd

from hotel_cancellation.calibration import calibration_summary
from hotel_cancellation.contracts import FEATURES
from hotel_cancellation.model import (
    ModelVersionError,
    load_model_bundle,
    local_contributions,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.bundle = load_model_bundle(ROOT / "artifacts")
            cls.pipeline, cls.metadata = cls.bundle.pipeline, cls.bundle.metadata
        except ModelVersionError as error:
            # CI installs the pinned version, so a mismatch there is a real failure.
            if os.environ.get("CI"):
                raise
            raise unittest.SkipTest(str(error)) from error

    def example_booking(self) -> pd.DataFrame:
        values = {
            **self.metadata["numeric_defaults"],
            **self.metadata["categorical_defaults"],
        }
        values["adults"] = max(1, int(values["adults"]))
        values["stays_in_week_nights"] = max(1, int(values["stays_in_week_nights"]))
        values["adr"] = max(0.0, float(values["adr"]))
        return pd.DataFrame([values], columns=FEATURES)

    def test_bundle_scores_valid_booking(self):
        scores, probabilities = self.bundle.score(self.example_booking())
        for value in (scores[0], probabilities[0]):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_calibrator_preserves_ranking(self):
        grid = np.linspace(0, 1, 501)
        calibrated = self.bundle.calibrator.predict(grid)
        self.assertTrue(np.all(np.diff(calibrated) >= 0))

    def test_committed_holdout_matches_artifact_scores(self):
        holdout = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv.gz")
        np.testing.assert_allclose(
            self.bundle.calibrator.predict(holdout["risk_score"].to_numpy()),
            holdout["cancellation_probability"].to_numpy(),
        )
        evaluation = holdout[holdout["calibration_role"] == "evaluation"]
        summary = calibration_summary(evaluation["is_canceled"], evaluation["cancellation_probability"])
        for metric, value in self.metadata["calibration"]["evaluation_calibrated"].items():
            self.assertAlmostEqual(summary[metric], value, places=10)

    def test_bundle_returns_local_contributions(self):
        contributions = local_contributions(self.pipeline, self.example_booking(), limit=5)
        self.assertEqual(len(contributions), 5)
        self.assertEqual(
            set(contributions.columns),
            {"feature", "contribution", "direction"},
        )


if __name__ == "__main__":
    unittest.main()
