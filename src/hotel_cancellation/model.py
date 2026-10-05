"""Model construction, prediction, and local contribution utilities."""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_cancellation.contracts import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class ModelVersionError(ValueError):
    """Raised when the installed scikit-learn differs from the one that exported the artifact."""


def build_pipeline() -> Pipeline:
    """Build the exact compact classification pipeline used for deployment."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def file_sha256(path: Path) -> str:
    """Return a SHA-256 fingerprint for a local model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline(path: Path, expected_sha256: str | None = None) -> Pipeline:
    """Load a fitted pipeline after optional integrity verification."""
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Model artifact integrity check failed")
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Model artifact is not a scikit-learn Pipeline")
    return pipeline


def load_model_bundle(artifact_dir: Path) -> tuple[Pipeline, dict]:
    """Load the validated model and its feature/validation metadata."""
    metadata_path = artifact_dir / "model_metadata.json"
    model_path = artifact_dir / "cancellation_logistic.joblib"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("features") != FEATURES:
        raise ValueError("Model metadata feature contract does not match the package")
    required = metadata.get("scikit_learn_version")
    if required != sklearn.__version__:
        raise ModelVersionError(
            f"Model artifact requires scikit-learn {required}; found {sklearn.__version__}. "
            "Install the pinned environment with `python -m pip install -r requirements.txt`, "
            "or rebuild the artifact for your version with `python scripts/download_data.py` "
            "followed by `python scripts/train_decision_artifacts.py`."
        )
    pipeline = load_pipeline(model_path, metadata.get("artifact_sha256"))
    return pipeline, metadata


def validate_booking(frame: pd.DataFrame) -> None:
    """Validate one or more inference rows before scoring."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    if (frame["adr"] < 0).any():
        raise ValueError("ADR must be non-negative")
    guests = frame[["adults", "children", "babies"]].fillna(0).sum(axis=1)
    if (guests <= 0).any():
        raise ValueError("At least one guest is required")
    nights = frame[["stays_in_weekend_nights", "stays_in_week_nights"]].sum(axis=1)
    if (nights <= 0).any():
        raise ValueError("At least one stay night is required")


def cancellation_probability(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return cancellation probabilities after validating the feature contract."""
    validate_booking(frame)
    return pipeline.predict_proba(frame[FEATURES])[:, 1]


def local_contributions(pipeline: Pipeline, frame: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Return the largest signed logistic contributions for a single booking."""
    if len(frame) != 1:
        raise ValueError("Local contributions require exactly one booking")
    transformed = pipeline.named_steps["prep"].transform(frame[FEATURES])
    coefficients = pipeline.named_steps["model"].coef_[0]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    values = np.asarray(transformed)[0] * coefficients
    result = pd.DataFrame({"feature": names, "contribution": values})
    result["direction"] = np.where(
        result["contribution"] >= 0,
        "higher cancellation score",
        "lower cancellation score",
    )
    return result.reindex(result["contribution"].abs().sort_values(ascending=False).index).head(limit)
