"""Model construction, prediction, and local contribution utilities."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_cancellation.contracts import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


MODEL_FILE = "cancellation_logistic.joblib"
CALIBRATOR_FILE = "cancellation_calibrator.joblib"


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
                    # A tight tolerance lets lbfgs converge fully, so refits agree across
                    # BLAS builds and thread counts instead of stopping at a build-specific point.
                    max_iter=10000,
                    tol=1e-8,
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


def load_artifact(path: Path, expected_sha256: str | None, expected_type: type):
    """Load a fitted artifact after integrity verification."""
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError(f"Artifact integrity check failed: {path.name}")
    artifact = joblib.load(path)
    if not isinstance(artifact, expected_type):
        raise TypeError(f"{path.name} is not a {expected_type.__name__}")
    return artifact


def load_pipeline(path: Path, expected_sha256: str | None = None) -> Pipeline:
    """Load a fitted pipeline after optional integrity verification."""
    return load_artifact(path, expected_sha256, Pipeline)


def fit_calibrator(scores: np.ndarray, outcomes: np.ndarray) -> IsotonicRegression:
    """Monotone map from the class-weighted risk score to an observed cancellation rate."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, outcomes)


@dataclass(frozen=True)
class ModelBundle:
    """The ranking pipeline, its isotonic calibrator, and their shared metadata."""

    pipeline: Pipeline
    calibrator: IsotonicRegression
    metadata: dict

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (risk scores for ranking and thresholds, calibrated probabilities)."""
        scores = risk_score(self.pipeline, frame)
        return scores, self.calibrator.predict(scores)


def load_model_bundle(artifact_dir: Path) -> ModelBundle:
    """Load the validated model, its calibrator and their feature/validation metadata."""
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
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
    pipeline = load_pipeline(artifact_dir / MODEL_FILE, metadata.get("artifact_sha256"))
    calibrator = load_artifact(
        artifact_dir / CALIBRATOR_FILE, metadata.get("calibrator_sha256"), IsotonicRegression
    )
    return ModelBundle(pipeline, calibrator, metadata)


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


def risk_score(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return the class-weighted logistic risk score after validating the feature contract.

    ``class_weight="balanced"`` inflates these scores above observed cancellation rates.
    Use them to rank bookings and apply the documented threshold; use the calibrated
    probability for anything that multiplies a probability by money.
    """
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
