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
- The sample-size table in `docs/experiment-design.md` uses the committed holdout: 12,929 flagged bookings at the 0.50 threshold, 46.76% cancelled, about 392 flagged bookings per week. These are planning figures, not a measured effect.
- A scikit-learn version mismatch now raises `ModelVersionError` with the fix. Outside CI the artifact tests skip with that message; in CI (pinned 1.8.0) they still run and fail on a mismatch.
- With the pinned CI dependencies: 8 tests pass, Ruff passes and the evidence check passes.

## Scoring API and experiment tracking — 2026-10-08

- Added a FastAPI service (`src/hotel_cancellation/api.py`) over the committed artifact. 8 new API tests check that API probabilities equal direct artifact scores to 12 decimal places. They also cover batch alignment, economics, unseen-category reporting, rejection of invalid bookings and unknown fields, and the batch-size limit.
- Built the Docker image and ran the container. `/health` returned the artifact version `b363eabf6c7c`, `/predict` returned a valid score, and the Docker health check reported `healthy`. CI now repeats the build and smoke test.
- With the pinned dependencies on Python 3.12: 16 tests pass, Ruff passes and the evidence check passes.
- Ran `scripts/track_experiments.py` on Python 3.12 / scikit-learn 1.8.0 against the fingerprinted source. Random Forest and Prior baseline reproduce the committed holdout metrics exactly.
- A Logistic Regression refit differs from the committed artifact by up to 0.007 in individual probabilities (+0.00005 holdout AP). The cause is lbfgs tolerance under a different BLAS build and thread count. The served artifact is unchanged. The gap is logged per run.
- Model selection used an inner chronological validation window (47,318 / 15,962 rows, inner cutoff 2016-06-23). The holdout did not select.
- Loaded `models:/hotel-cancellation-risk@production` back from the MLflow registry and scored rows as probabilities.
