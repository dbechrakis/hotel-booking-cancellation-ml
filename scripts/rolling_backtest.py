"""Score the deployed pipeline at four chronological cutoffs, as fixed in VALIDATION.md before the run.

Each cutoff trains on matured bookings before it and tests on the bookings made in the next six months. As in the deployed
system, the earliest 30% of the test window sets the isotonic calibrator and the 80%-recall threshold and the other 70% evaluates.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from hotel_cancellation.calibration import calibration_summary
from hotel_cancellation.data import feature_target, source_sha256, split_at
from hotel_cancellation.model import build_pipeline, fit_calibrator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
CUTOFFS = ("2016-04-12", "2016-07-12", "2016-10-12", "2017-01-12")
CALIBRATION_SHARE = 0.3
TARGET_RECALL = 0.80
MIN_TEST_BOOKINGS = 5000
OUTPUT_CSV = ROOT / "outputs" / "rolling_backtest.csv"
OUTPUT_JSON = ROOT / "outputs" / "rolling_backtest.json"


def recall_threshold(scores, outcomes, target_recall: float) -> float:
    positive_scores = np.sort(scores[outcomes == 1])[::-1]
    return float(positive_scores[int(np.ceil(target_recall * len(positive_scores))) - 1])


def score_cutoff(raw: pd.DataFrame, cutoff: str) -> dict:
    split = split_at(raw, cutoff)
    if int(split.test_mask.sum()) < MIN_TEST_BOOKINGS:
        raise ValueError(f"{cutoff}: fewer than {MIN_TEST_BOOKINGS} test bookings")
    features, target = feature_target(raw)
    pipeline = build_pipeline()
    pipeline.fit(features.loc[split.train_mask], target.loc[split.train_mask])
    test_target = target.loc[split.test_mask].to_numpy()
    scores = pipeline.predict_proba(features.loc[split.test_mask])[:, 1]
    booking_date = split.booking_date.loc[split.test_mask]
    edge = booking_date.sort_values().iloc[int(len(booking_date) * CALIBRATION_SHARE)]
    cal = (booking_date < edge).to_numpy()
    ev = ~cal
    calibrator = fit_calibrator(scores[cal], test_target[cal])
    threshold = recall_threshold(scores[cal], test_target[cal], TARGET_RECALL)
    flagged = scores[ev] >= threshold
    summary = calibration_summary(test_target[ev], calibrator.predict(scores[ev]))
    return {"cutoff": cutoff, "train_rows": int(split.train_mask.sum()), "test_rows": int(split.test_mask.sum()), "evaluation_rows": int(ev.sum()),
            "average_precision": float(average_precision_score(test_target, scores)), "roc_auc": float(roc_auc_score(test_target, scores)),
            "brier": summary["brier"], "ece": summary["ece"], "threshold": threshold, "flagged_share": float(flagged.mean()),
            "recall": float(recall_score(test_target[ev], flagged)), "precision": float(precision_score(test_target[ev], flagged)),
            "calibration_cancellation_rate": float(test_target[cal].mean()), "evaluation_cancellation_rate": float(test_target[ev].mean())}


def main() -> list[dict]:
    if source_sha256(SOURCE) != EXPECTED_SOURCE_SHA:
        raise ValueError("Source fingerprint does not match the validated dataset")
    raw = pd.read_csv(SOURCE)
    rows = [score_cutoff(raw, c) for c in CUTOFFS]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
    return rows


if __name__ == "__main__":
    print(pd.DataFrame(main()).round(4).to_string())
