# Validation record

Executed the complete revised notebook against the 119,390-row public source, with its SHA-256 recorded in `outputs/validation.json`. Python cells were executed sequentially in a single process because kernel sockets were unavailable. All cell outputs are saved.

The split uses inferred booking date and fully matured training stays. Training: 79,590; test: 23,989; purged: 15,811; cutoff: 2017-01-12.

Old random-split scores and office documents are superseded. No repeated temporal cross-validation, live intervention or production deployment was performed.

## Decision-system verification — 2026-09-22

- Re-downloaded the source and verified its SHA-256 fingerprint.
- Reproduced the exact committed Logistic Regression accuracy, precision, recall, F1, average precision, and ROC AUC under the pinned environment.
- Exported a 5 KB scikit-learn pipeline and verified its SHA-256 before inference.
- Exported 23,989 aligned later-holdout probabilities for threshold simulation.
- Added tests for risk bands, intervention economics, threshold counts, artifact loading, inference, and local contributions.
- Streamlit AppTest started the complete application without exceptions.
- Streamlit AppTest submitted the default booking and produced probability, risk band, expected value, and contribution outputs.

The app is a portfolio decision-support demonstration. No intervention success or realised revenue uplift was observed or inferred from the source data.

## Experiment sizing and version guard — 2026-10-05

- Added `experiment_sample_size()` (two-sided two-proportion z-test). It reproduces the textbook 388 bookings per arm for 50% → 40% at α = 0.05 and 80% power.
- The sample-size table in `docs/experiment-design.md` uses the committed holdout flagged population and its cancellation rate. These are planning figures, not a measured effect.
- A scikit-learn version mismatch now raises `ModelVersionError` with the fix. Outside CI the artifact tests skip with that message; in CI (pinned 1.8.0) they still run and fail on a mismatch.
- With the pinned CI dependencies: 8 tests pass, Ruff passes and the evidence check passes. (The sample-size figures above were updated on 2026-10-08 for the converged model: 13,221 flagged, 46.41% cancelled, about 401 per week.)

## Scoring API and experiment tracking — 2026-10-08

- Added a FastAPI service (`src/hotel_cancellation/api.py`) over the committed artifact. 8 new API tests check that API probabilities equal direct artifact scores to 12 decimal places. They also cover batch alignment, economics, unseen-category reporting, rejection of invalid bookings and unknown fields, and the batch-size limit.
- Built the Docker image and ran the container. `/health` returned the artifact version `b363eabf6c7c`, `/predict` returned a valid score, and the Docker health check reported `healthy`. CI now repeats the build and smoke test.
- With the pinned dependencies on Python 3.12: 16 tests pass, Ruff passes and the evidence check passes.
- Ran `scripts/track_experiments.py` on Python 3.12 / scikit-learn 1.8.0 against the fingerprinted source. Every candidate reproduces the committed holdout metrics.
- Model selection used an inner chronological validation window (47,318 / 15,962 rows, inner cutoff 2016-06-23). The holdout did not select.
- Loaded `models:/hotel-cancellation-risk@production` back from the MLflow registry and scored rows as probabilities.

## Converged logistic fit and isotonic calibration — 2026-10-08

- The logistic model now trains with `tol=1e-8` (`max_iter=10000`). The notebook was re-executed end to end with this setting. Logistic Regression holdout metrics: accuracy 0.6453, precision 0.4641, recall 0.8115, F1 0.5905, AP 0.6444, ROC AUC 0.7750. Prior baseline and Random Forest are unchanged.
- `train_decision_artifacts.py` reproduces the notebook's metrics within its 1e-6 tolerance. Refits on Python 3.12 and 3.13, with one and four BLAS threads, agree to about 1e-5 in individual probabilities.
- Added an isotonic calibrator, fitted on the earliest 30% of later bookings (7,154, before 2017-02-08) and evaluated on the other 16,835. Brier 0.229 → 0.174, ECE 0.221 → 0.013, mean predicted 51.2% → 30.1% against 29.2% observed.
- A first attempt calibrated on an inner window inside the training period. It worsened calibration on later bookings (49.5% predicted vs 31.5% observed) because `arrival_date_year` extrapolates. It was not adopted; see `docs/calibration.md`.
- The API, the Streamlit app and the threshold simulator now separate the risk score (ranking and threshold) from the calibrated probability (risk bands and expected value).
- With pinned dependencies on Python 3.12: 22 tests pass, Ruff passes and the evidence check passes. A Streamlit AppTest scored the default booking without exceptions: risk score 0.88, calibrated probability 60.1%.

## Removed arrival_date_year; review threshold chosen by rule — 2026-10-08

- `arrival_date_year` is no longer a model input; the contract now has 21 features. As a numeric input it extrapolated beyond the training years and inflated later-booking scores by about 14 points.
- The notebook was re-executed without the feature.
  - Logistic Regression: AP 0.6457, ROC AUC 0.7753, and F1 0.5435 at the fixed 0.50 comparison threshold.
  - Random Forest: AP 0.6797, ROC AUC 0.8080.
  - The lead-time regression improved: Ridge R² −1.83 → +0.06, gradient boosting 0.15 → 0.29.
- The review threshold is now chosen as the highest risk score reaching 80% recall on the calibration window (the first 30% of later bookings). The result is 0.372. On the 16,835 evaluation bookings it flagged 44.8% and reached 69.6% recall at 45.3% precision, short of the target.
- The isotonic calibrator was refitted on the same window. On evaluation bookings, Brier fell 0.182 → 0.174 and ECE 0.073 → 0.013.
- The API and the app no longer take an arrival year. Their flag and the app's default thresholds use the policy threshold.
- With pinned dependencies on Python 3.12: 22 tests pass, Ruff passes and the evidence check passes. A Streamlit AppTest scored the default booking without exceptions (risk score 0.78, calibrated probability 55.8%).
