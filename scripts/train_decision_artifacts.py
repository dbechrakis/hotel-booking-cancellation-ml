"""Train and export the compact cancellation decision-system artifacts."""

from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hotel_cancellation.contracts import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from hotel_cancellation.data import chronological_split, feature_target, source_sha256
from hotel_cancellation.model import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ARTIFACT_DIR / "cancellation_logistic.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
PREDICTIONS_PATH = OUTPUT_DIR / "holdout_predictions.csv.gz"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    probabilities = pipeline.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(test_target, predictions),
        "precision": precision_score(test_target, predictions),
        "recall": recall_score(test_target, predictions),
        "f1": f1_score(test_target, predictions),
        "average_precision": average_precision_score(test_target, probabilities),
        "roc_auc": roc_auc_score(test_target, probabilities),
    }
    expected_metrics = pd.read_csv(OUTPUT_DIR / "classification_metrics.csv").set_index("model")
    expected = expected_metrics.loc["Logistic Regression", list(metrics)]
    for metric, value in metrics.items():
        if abs(value - float(expected[metric])) > 1e-10:
            raise ValueError(f"Reproduced {metric} does not match committed evidence")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    prediction_output = pd.DataFrame(
        {
            "row_id": test_features.index,
            "booking_date": split.booking_date.loc[split.test_mask].dt.date.astype(str),
            "hotel": raw.loc[split.test_mask, "hotel"].values,
            "market_segment": raw.loc[split.test_mask, "market_segment"].values,
            "lead_time": raw.loc[split.test_mask, "lead_time"].values,
            "adr": raw.loc[split.test_mask, "adr"].values,
            "is_canceled": test_target.values,
            "cancellation_probability": probabilities,
        }
    )
    prediction_output.to_csv(PREDICTIONS_PATH, index=False, compression="gzip")

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
        "scikit_learn_version": sklearn.__version__,
        "booking_date_cutoff": str(split.cutoff.date()),
        "train_rows": int(split.train_mask.sum()),
        "test_rows": int(split.test_mask.sum()),
        "purged_rows": int(split.purged_mask.sum()),
        "threshold": 0.5,
        "features": FEATURES,
        "holdout_metrics": metrics,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "category_options": category_options,
        "limitations": [
            "one later historical holdout",
            "probabilities are not claimed to be perfectly calibrated",
            "intervention effectiveness is not observed in this dataset",
            "SHAP and logistic contributions are associative, not causal",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"Saved {PREDICTIONS_PATH} ({len(prediction_output):,} rows)")
    print(pd.Series(metrics).round(4).to_string())


if __name__ == "__main__":
    main()
