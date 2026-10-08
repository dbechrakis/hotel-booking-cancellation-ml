"""Compare candidate models under MLflow with selection kept off the holdout.

Each candidate is fitted on an inner chronological split of the training period and
selected on that inner validation window only. Every candidate is then refitted on the
full matured training period and scored once on the later holdout, so the holdout
reports performance and never chooses the model.

    python scripts/download_data.py
    python scripts/track_experiments.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path
import json
import os
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from hotel_cancellation.contracts import FEATURES
from hotel_cancellation.data import chronological_split, feature_target, source_sha256
from hotel_cancellation.model import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
EXPERIMENT = "hotel-cancellation"
REGISTERED_MODEL = "hotel-cancellation-risk"
SELECTION_METRIC = "average_precision"
DEPLOYED = "Logistic Regression"
# MLflow serialises with skops. These are the reviewed non-default types our own
# fitted candidates contain; they are trusted because this script created them.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def candidate(estimator) -> Pipeline:
    """Reuse the deployed preprocessing so candidates differ only in the estimator."""
    pipeline = build_pipeline()
    pipeline.steps[-1] = ("model", estimator)
    return pipeline


CANDIDATES = {
    "Prior baseline": candidate(DummyClassifier(strategy="prior")),
    "Logistic Regression": build_pipeline(),
    "Random Forest": candidate(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    ),
    "Hist Gradient Boosting": candidate(
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
    ),
}


def evaluate(target: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "average_precision": average_precision_score(target, probabilities),
        "roc_auc": roc_auc_score(target, probabilities),
        "brier": brier_score_loss(target, probabilities),
    }


def git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def calibration_table(target: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    observed, predicted = calibration_curve(target, probabilities, n_bins=10, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError("Run python scripts/download_data.py first")
    source_hash = source_sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA:
        raise ValueError("Source fingerprint does not match the validated dataset")

    raw = pd.read_csv(SOURCE)
    features, target = feature_target(raw)
    outer = chronological_split(raw)
    # The inner split applies the same matured-label contract inside the training period.
    training_period = raw.loc[outer.train_mask].reset_index(drop=True)
    inner = chronological_split(training_period)
    inner_features, inner_target = feature_target(training_period)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    )
    mlflow.set_experiment(EXPERIMENT)
    committed = pd.read_csv(ROOT / "outputs" / "classification_metrics.csv").set_index("model")

    rows = []
    with mlflow.start_run(run_name="chronological-model-comparison") as parent:
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "source_sha256": source_hash,
                "selection_metric": f"validation_{SELECTION_METRIC}",
                "scikit_learn_version": sklearn.__version__,
            }
        )
        mlflow.log_params(
            {
                "outer_cutoff": str(outer.cutoff.date()),
                "inner_cutoff": str(inner.cutoff.date()),
                "inner_train_rows": int(inner.train_mask.sum()),
                "validation_rows": int(inner.test_mask.sum()),
                "train_rows": int(outer.train_mask.sum()),
                "holdout_rows": int(outer.test_mask.sum()),
                "n_features": len(FEATURES),
            }
        )
        for name, template in CANDIDATES.items():
            with mlflow.start_run(run_name=name, nested=True):
                model_params = template.named_steps["model"].get_params()
                mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})

                validation_model = clone(template).fit(
                    inner_features.loc[inner.train_mask], inner_target.loc[inner.train_mask]
                )
                validation = evaluate(
                    inner_target.loc[inner.test_mask],
                    validation_model.predict_proba(inner_features.loc[inner.test_mask])[:, 1],
                )

                final_model = clone(template).fit(
                    features.loc[outer.train_mask], target.loc[outer.train_mask]
                )
                holdout_probabilities = final_model.predict_proba(features.loc[outer.test_mask])[:, 1]
                holdout = evaluate(target.loc[outer.test_mask], holdout_probabilities)

                mlflow.log_metrics({f"validation_{k}": v for k, v in validation.items()})
                mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})
                if name in committed.index:
                    # Records any drift from the committed notebook metrics (about 1e-7 now
                    # that the logistic fit converges to tol=1e-8).
                    mlflow.log_metric(
                        "holdout_ap_delta_vs_committed",
                        holdout["average_precision"] - committed.loc[name, "average_precision"],
                    )
                mlflow.log_table(
                    calibration_table(target.loc[outer.test_mask], holdout_probabilities),
                    "holdout_calibration.json",
                )
                example = features.loc[outer.train_mask].head(5)
                info = mlflow.sklearn.log_model(
                    final_model,
                    name="model",
                    input_example=example,
                    signature=infer_signature(example, final_model.predict_proba(example)[:, 1]),
                    registered_model_name=REGISTERED_MODEL if name == DEPLOYED else None,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                    pyfunc_predict_fn="predict_proba",
                )
                rows.append(
                    {
                        "model": name,
                        **{f"validation_{k}": v for k, v in validation.items()},
                        **{f"holdout_{k}": v for k, v in holdout.items()},
                    }
                )
                if name == DEPLOYED:
                    deployed_version = info.registered_model_version

        summary = pd.DataFrame(rows)
        selected = summary.loc[summary[f"validation_{SELECTION_METRIC}"].idxmax(), "model"]
        mlflow.set_tag("selected_on_validation", selected)
        mlflow.set_tag("deployed_model", DEPLOYED)

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "production", deployed_version)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "parent_run_id": parent.info.run_id,
                "git_commit": git_commit(),
                "source_sha256": source_hash,
                "scikit_learn_version": sklearn.__version__,
                "outer_cutoff": str(outer.cutoff.date()),
                "inner_cutoff": str(inner.cutoff.date()),
                "inner_train_rows": int(inner.train_mask.sum()),
                "validation_rows": int(inner.test_mask.sum()),
                "train_rows": int(outer.train_mask.sum()),
                "holdout_rows": int(outer.test_mask.sum()),
                "selection_metric": f"validation_{SELECTION_METRIC}",
                "selected_on_validation": selected,
                "deployed_model": DEPLOYED,
                "registered_model": f"{REGISTERED_MODEL}@production (version {deployed_version})",
            },
            indent=2,
        )
    )
    columns = ["model", "validation_average_precision", "holdout_average_precision",
               "holdout_roc_auc", "holdout_f1", "holdout_brier"]
    print(summary[columns].round(4).to_string(index=False))
    print(f"Selected on validation: {selected}; deployed: {DEPLOYED}")


if __name__ == "__main__":
    main()
