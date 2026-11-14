"""Score the deployed pipeline with and without the ``country`` input on the committed chronological split.

``country`` is often confirmed only at check-in, so a cancelled booking can keep the hotel's default country. This script
measures what the ranking, the calibration and the review threshold lose when the input is left out. Both variants use the
same rows, the same split, the same calibration window (the earliest 30% of later bookings) and the same recall rule.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from hotel_cancellation.calibration import calibration_summary
from hotel_cancellation.contracts import NOTEBOOK_CATEGORICAL_FEATURES, NOTEBOOK_FEATURES, NUMERIC_FEATURES
from hotel_cancellation.data import chronological_split, feature_target, source_sha256
from hotel_cancellation.model import build_pipeline, fit_calibrator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
CALIBRATION_SHARE = 0.3
TARGET_RECALL = 0.80
OUTPUT = ROOT / "outputs" / "country_check.json"


def recall_threshold(scores, outcomes, target_recall: float) -> float:
    positive_scores = np.sort(scores[outcomes == 1])[::-1]
    return float(positive_scores[int(np.ceil(target_recall * len(positive_scores))) - 1])


def score_variant(raw: pd.DataFrame, split, drop_country: bool) -> dict:
    features, target = feature_target(raw, NOTEBOOK_FEATURES)
    categorical = [c for c in NOTEBOOK_CATEGORICAL_FEATURES if not (drop_country and c == "country")]
    use = [c for c in NOTEBOOK_FEATURES if not (drop_country and c == "country")]
    pipeline = build_pipeline(NUMERIC_FEATURES, categorical)
    pipeline.fit(features.loc[split.train_mask, use], target.loc[split.train_mask])
    test_target = target.loc[split.test_mask].to_numpy()
    scores = pipeline.predict_proba(features.loc[split.test_mask, use])[:, 1]
    booking_date = split.booking_date.loc[split.test_mask]
    cutoff = booking_date.sort_values().iloc[int(len(booking_date) * CALIBRATION_SHARE)]
    cal = (booking_date < cutoff).to_numpy()
    ev = ~cal
    calibrator = fit_calibrator(scores[cal], test_target[cal])
    threshold = recall_threshold(scores[cal], test_target[cal], TARGET_RECALL)
    flagged = scores[ev] >= threshold
    probabilities = calibrator.predict(scores[ev])
    return {"roc_auc": float(roc_auc_score(test_target, scores)), "average_precision": float(average_precision_score(test_target, scores)),
            "calibration_evaluation": calibration_summary(test_target[ev], probabilities),
            "threshold": threshold, "evaluation_flagged_share": float(flagged.mean()),
            "evaluation_recall": float(recall_score(test_target[ev], flagged)), "evaluation_precision": float(precision_score(test_target[ev], flagged)),
            "evaluation_rows": int(ev.sum())}


def main() -> dict:
    if source_sha256(SOURCE) != EXPECTED_SOURCE_SHA:
        raise ValueError("Source fingerprint does not match the validated dataset")
    raw = pd.read_csv(SOURCE)
    split = chronological_split(raw)
    portugal = raw.loc[split.test_mask, "country"].eq("PRT")
    result = {"with_country": score_variant(raw, split, False), "without_country": score_variant(raw, split, True),
              "portugal_share_of_cancellations_later": float(portugal[raw.loc[split.test_mask, "is_canceled"] == 1].mean())}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
