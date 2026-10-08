"""Train and export the compact cancellation decision-system artifacts.

Exports the ranking pipeline, an isotonic calibrator and a review threshold. The
calibrator and the threshold are both chosen on the earliest 30% of post-cutoff bookings
and evaluated on the later 70%, mirroring how a deployed score would be recalibrated and
re-thresholded on its most recent matured outcomes. The later 70% is never used to choose.
"""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hotel_cancellation.calibration import calibration_summary, reliability_table  # noqa: E402
from hotel_cancellation.contracts import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402
from hotel_cancellation.data import chronological_split, feature_target, source_sha256  # noqa: E402
from hotel_cancellation.model import (  # noqa: E402
    CALIBRATOR_FILE,
    MODEL_FILE,
    build_pipeline,
    file_sha256,
    fit_calibrator,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ARTIFACT_DIR / MODEL_FILE
CALIBRATOR_PATH = ARTIFACT_DIR / CALIBRATOR_FILE
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
PREDICTIONS_PATH = OUTPUT_DIR / "holdout_predictions.csv.gz"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
METRIC_THRESHOLD = 0.5  # the notebook's fixed comparison threshold, used to reproduce it
TARGET_RECALL = 0.80  # policy: the review queue should catch four in five cancellations
# The logistic fit converges to tol=1e-8, so refits on other BLAS builds agree with the
# notebook's committed metrics to about 1e-7; a looser fit drifted by up to 1e-3.
REPRODUCTION_TOLERANCE = 1e-6
CALIBRATION_SHARE = 0.3


def recall_threshold(scores, outcomes, target_recall: float) -> float:
    """Highest score threshold whose flagged set still catches ``target_recall`` of positives."""
    positive_scores = np.sort(scores[outcomes == 1])[::-1]
    needed = int(np.ceil(target_recall * len(positive_scores)))
    return float(positive_scores[needed - 1])


def plot_reliability(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8, label="Perfect calibration")
    for label, table in tables.items():
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=label)
    ax.set_xlabel("Mean predicted cancellation probability")
    ax.set_ylabel("Observed cancellation rate")
    ax.set_title("Reliability on later bookings (evaluation 70%, 10 equal-count bins)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("Run python scripts/download_data.py first")
    source_hash = source_sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA:
        raise ValueError("Source fingerprint does not match the validated dataset")

    raw = pd.read_csv(SOURCE)
    split = chronological_split(raw)
    features, target = feature_target(raw)

    pipeline = build_pipeline()
    pipeline.fit(features.loc[split.train_mask], target.loc[split.train_mask])
    test_features = features.loc[split.test_mask]
    test_target = target.loc[split.test_mask]
    scores = pipeline.predict_proba(test_features)[:, 1]
    predictions = (scores >= METRIC_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(test_target, predictions),
        "precision": precision_score(test_target, predictions),
        "recall": recall_score(test_target, predictions),
        "f1": f1_score(test_target, predictions),
        "average_precision": average_precision_score(test_target, scores),
        "roc_auc": roc_auc_score(test_target, scores),
    }
    expected_metrics = pd.read_csv(OUTPUT_DIR / "classification_metrics.csv").set_index("model")
    expected = expected_metrics.loc["Logistic Regression", list(metrics)]
    for metric, value in metrics.items():
        if abs(value - float(expected[metric])) > REPRODUCTION_TOLERANCE:
            raise ValueError(
                f"Reproduced {metric} {value:.8f} differs from committed {float(expected[metric]):.8f}"
            )

    # Forward recalibration: fit on the earliest post-cutoff bookings, evaluate on the rest.
    booking_date = split.booking_date.loc[split.test_mask]
    calibration_cutoff = booking_date.sort_values().iloc[int(len(booking_date) * CALIBRATION_SHARE)]
    calibration_rows = (booking_date < calibration_cutoff).to_numpy()
    evaluation_rows = ~calibration_rows
    calibration_target = test_target.to_numpy()[calibration_rows]
    calibrator = fit_calibrator(scores[calibration_rows], calibration_target)
    threshold = recall_threshold(scores[calibration_rows], calibration_target, TARGET_RECALL)
    evaluation_flagged = scores[evaluation_rows] >= threshold
    evaluation_truth = test_target.to_numpy()[evaluation_rows]
    policy = {
        "rule": f"highest risk-score threshold reaching {TARGET_RECALL:.0%} recall on the calibration window",
        "threshold": threshold,
        "evaluation_flagged_share": float(evaluation_flagged.mean()),
        "evaluation_precision": precision_score(evaluation_truth, evaluation_flagged),
        "evaluation_recall": recall_score(evaluation_truth, evaluation_flagged),
        "evaluation_f1": f1_score(evaluation_truth, evaluation_flagged),
        "holdout_flagged": int((scores >= threshold).sum()),
        "holdout_flagged_cancellation_rate": float(test_target.to_numpy()[scores >= threshold].mean()),
    }
    probabilities = calibrator.predict(scores)
    evaluation_target = test_target.to_numpy()[evaluation_rows]

    calibration = {
        "method": "isotonic regression on the class-weighted logistic risk score",
        "fitted_on": f"earliest {CALIBRATION_SHARE:.0%} of holdout bookings by booking date",
        "calibration_cutoff": str(calibration_cutoff.date()),
        "calibration_rows": int(calibration_rows.sum()),
        "evaluation_rows": int(evaluation_rows.sum()),
        "calibration_window_cancellation_rate": float(test_target.to_numpy()[calibration_rows].mean()),
        "evaluation_uncalibrated": calibration_summary(evaluation_target, scores[evaluation_rows]),
        "evaluation_calibrated": calibration_summary(evaluation_target, probabilities[evaluation_rows]),
    }
    raw_table = reliability_table(evaluation_target, scores[evaluation_rows])
    calibrated_table = reliability_table(evaluation_target, probabilities[evaluation_rows])

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    joblib.dump(calibrator, CALIBRATOR_PATH, compress=3)

    prediction_output = pd.DataFrame(
        {
            "row_id": test_features.index,
            "booking_date": split.booking_date.loc[split.test_mask].dt.date.astype(str),
            "hotel": raw.loc[split.test_mask, "hotel"].values,
            "market_segment": raw.loc[split.test_mask, "market_segment"].values,
            "lead_time": raw.loc[split.test_mask, "lead_time"].values,
            "adr": raw.loc[split.test_mask, "adr"].values,
            "is_canceled": test_target.values,
            "risk_score": scores,
            "cancellation_probability": probabilities,
            "calibration_role": ["calibration" if row else "evaluation" for row in calibration_rows],
        }
    )
    prediction_output.to_csv(PREDICTIONS_PATH, index=False, compression="gzip")
    pd.concat(
        [raw_table.assign(score="uncalibrated risk score"), calibrated_table.assign(score="isotonic probability")]
    ).to_csv(OUTPUT_DIR / "calibration_reliability.csv", index=False)
    plot_reliability(
        {"Uncalibrated risk score": raw_table, "Isotonic probability": calibrated_table},
        OUTPUT_DIR / "calibration_reliability.png",
    )

    training_frame = features.loc[split.train_mask]
    numeric_defaults = {
        feature: float(training_frame[feature].median()) for feature in NUMERIC_FEATURES
    }
    categorical_defaults = {
        feature: str(training_frame[feature].mode(dropna=True).iloc[0])
        for feature in CATEGORICAL_FEATURES
    }
    category_options = {
        feature: sorted(training_frame[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    metadata = {
        "model": "Logistic Regression",
        "model_purpose": "later-booking cancellation risk ranking and decision support",
        "source_sha256": source_hash,
        "artifact_sha256": file_sha256(MODEL_PATH),
        "calibrator_sha256": file_sha256(CALIBRATOR_PATH),
        "scikit_learn_version": sklearn.__version__,
        "booking_date_cutoff": str(split.cutoff.date()),
        "train_rows": int(split.train_mask.sum()),
        "test_rows": int(split.test_mask.sum()),
        "purged_rows": int(split.purged_mask.sum()),
        "threshold": threshold,
        "threshold_scale": "risk_score",
        "policy": policy,
        "features": FEATURES,
        "holdout_metrics": metrics,
        "calibration": calibration,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "category_options": category_options,
        "limitations": [
            "one later historical holdout",
            "calibration uses early post-cutoff outcomes; operationally they mature only after those stays",
            "intervention effectiveness is not observed in this dataset",
            "SHAP and logistic contributions are associative, not causal",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved {MODEL_PATH.name} and {CALIBRATOR_PATH.name}; {len(prediction_output):,} holdout rows")
    print(pd.Series(metrics).round(4).to_string())
    print(json.dumps(policy, indent=2))
    print(json.dumps({k: calibration[k] for k in ["evaluation_uncalibrated", "evaluation_calibrated"]}, indent=2))


if __name__ == "__main__":
    main()
