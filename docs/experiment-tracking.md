# Experiment tracking and model registry

`scripts/track_experiments.py` compares candidate models under MLflow without letting
the later holdout choose the winner.

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/download_data.py
python scripts/track_experiments.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Protocol

```mermaid
flowchart LR
    A["Matured training period<br/>(before 2017-01-12)"] --> B["Inner train<br/>47,318 rows"]
    A --> C["Inner validation<br/>15,962 rows (from 2016-06-23)"]
    B --> D["Fit each candidate"]
    D --> C
    C --> E["Select on validation AP"]
    A --> F["Refit every candidate"]
    F --> G["Score once on later holdout<br/>23,989 rows"]
```

1. The same matured chronological split used for the holdout is applied again **inside**
   the training period. Inner-train outcomes resolve before the inner cutoff.
2. Every candidate shares the deployed preprocessing; only the estimator changes.
3. Selection uses **validation average precision**, because the decision is a ranking
   of which bookings to review.
4. Every candidate is refitted on the full training period and scored once on the
   holdout. Holdout numbers are reported, never used to choose.
5. MLflow records parameters, validation and holdout metrics (including Brier score),
   a holdout calibration table, the fitted pipeline with its input signature, the source
   SHA-256 and the git commit. The deployed logistic pipeline is registered as
   `hotel-cancellation-risk` with the alias `production`.

## Results (Python 3.12, scikit-learn 1.8.0)

Committed in [`outputs/experiments/model_comparison.csv`](../outputs/experiments/model_comparison.csv)
with the run manifest in [`run_manifest.json`](../outputs/experiments/run_manifest.json).

| Model | Validation AP | Holdout AP | Holdout ROC AUC | Holdout F1 @0.50 | Holdout Brier |
|---|---:|---:|---:|---:|---:|
| Prior baseline | 0.269 | 0.315 | 0.500 | 0.000 | 0.218 |
| **Logistic Regression** (deployed) | 0.471 | 0.646 | 0.775 | 0.544 | 0.175 |
| **Random Forest** (selected on validation) | **0.502** | 0.680 | 0.808 | 0.569 | 0.163 |
| Hist Gradient Boosting | 0.449 | **0.690** | **0.817** | **0.618** | **0.161** |

## What the comparison shows

- **Validation and holdout disagree on the winner.** Random Forest leads on validation;
  gradient boosting has the *lowest* validation AP of the learned models but the best
  holdout AP. Choosing on the holdout would have reported 0.690 AP for a model picked
  because it scored 0.690, an optimistic estimate. The protocol keeps that choice honest,
  and the disagreement itself shows that the ranking between tree models is not stable
  across periods.
- **The deployed logistic model is the weakest learned ranker, and its raw score is not a
  probability.** `class_weight="balanced"` inflates scores, so the deployed system pairs
  the score with an isotonic calibrator; see [calibration](calibration.md).
- **Why it is still deployed:** compact artifact and exact signed contributions. Replacing
  it with a tree model is a product decision that would need its own calibration and a
  second temporal backtest.
- These numbers exclude `arrival_date_year`. With it, every model was fitted on a numeric
  year that later bookings exceed. Logistic Regression's holdout Brier score was 0.218
  then, versus 0.175 now.

## Reproducibility

The logistic fit uses `tol=1e-8`, so lbfgs converges fully (about 1,100 iterations)
instead of stopping at a point that depends on the BLAS build and thread count. Refits on
Python 3.12 and 3.13, single- and multi-threaded, agree to about 1e-5 in individual
probabilities and 1e-7 in holdout metrics, and every candidate reproduces the committed
metrics (`holdout_ap_delta_vs_committed` = 0).
