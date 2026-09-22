from pathlib import Path
import unittest

import pandas as pd

from hotel_cancellation.contracts import FEATURES
from hotel_cancellation.model import (
    cancellation_probability,
    load_model_bundle,
    local_contributions,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.metadata = load_model_bundle(ROOT / "artifacts")

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
        probability = cancellation_probability(self.pipeline, self.example_booking())[0]
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_bundle_returns_local_contributions(self):
        contributions = local_contributions(self.pipeline, self.example_booking(), limit=5)
        self.assertEqual(len(contributions), 5)
        self.assertEqual(
            set(contributions.columns),
            {"feature", "contribution", "direction"},
        )


if __name__ == "__main__":
    unittest.main()
