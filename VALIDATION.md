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

## Score monitoring — 2026-10-08

- The API now logs one JSON line per scored booking (no booking attributes) and serves `GET /monitoring`. That endpoint reports PSI of a rolling 5,000-score window against the deciles of the calibration window, plus flagged share, mean probability, unseen-category share and latency. The reference profile is stored in `model_metadata.json`. Retraining reproduced both joblib artifacts byte for byte.
- 6 new tests cover PSI, drift bands, the rolling window, the API endpoint and log content. 28 tests pass and Ruff passes.
- `monitoring_replay.py` replayed the 16,835 evaluation bookings by booking week (27 weeks with at least 200 bookings):
  - The trailing-2,000-score PSI was ok in the first 2 weeks. From 2017-02-20 on, every week was a warning or an alert (21 alert weeks).
  - Realised recall was below 80% in 26 of 27 weeks.
  - PSI and recall correlate at −0.62.
- The shift is mostly non-refundable bookings: 13.1% of the calibration window, 1.4% of the evaluation bookings.

## Removed country from the deployed model, with the comparison — 2026-11-14

- The notebook's comparison, the Random Forest and the tracked experiments were run with `country` as an input. They are unchanged and recorded as history. The deployed contract now has 20 features and no longer accepts `country`.
- Reason: the guest's country is often confirmed only at check-in, so a cancelled booking can keep the hotel's default country. Portuguese bookings are 62% of cancellations and cancel at 56.6% against 23.6% for the rest. The data cannot separate leakage from a real domestic effect.
- `scripts/country_check.py` scores both variants on the same split with the same calibration window and recall rule (`outputs/country_check.json`). Without `country`: ROC-AUC 0.7753 → 0.7432, average precision 0.6457 → 0.6224, ECE after isotonic 0.013 → 0.025, recall at the threshold 69.6% → 66.3%, precision 45.3% → 42.9%.
- `scripts/train_decision_artifacts.py` first reproduces the committed Logistic Regression row with `country` (tolerance 1e-6), then fits the deployed model without it. The artifacts, the calibrator, the threshold (0.372 → 0.469), the holdout predictions and `outputs/monitoring/` were regenerated. The evidence check now compares the committed row with `notebook_reference_metrics` and the feature list with the notebook's list minus `removed_inputs`.
- Monitoring replay on the new model: the first alert is still 2017-02-20, 23 of 27 weeks are alerts, realised recall is below 80% in 27 of 27 weeks (it was 26), and PSI and recall correlate at −0.23 (it was −0.62). The shortfall is no longer predicted as well by the PSI alone.
- The API, the app, the example booking and the docs lost the field. A request that carries `country` is rejected (422). `make check` passes (38 tests, evidence check, lint).
- Not rerun: the MLflow tracking, the notebook, the Hugging Face deployment. The hosted app needs a redeploy to use the new artifacts.


## Rolling temporal backtest, design fixed before it was run — 2026-11-14

Written before any rolling cutoff was scored. The single chronological split of the README is one draw. This section fixes how a repeated backtest will be read.

- **Cutoffs.** 2016-04-12, 2016-07-12, 2016-10-12 and 2017-01-12 (the committed split's cutoff). Counts only, taken first: the later six months hold 23,412, 28,015, 31,627 and 22,509 bookings, so all four clear the 5,000 floor.
- **Training rows** at a cutoff: the committed rule. Booking date, scheduled departure and resolution date all before the cutoff.
- **Test window** at a cutoff: bookings made in the six months from the cutoff. Later bookings are purged.
- **Model:** the deployed logistic pipeline, unchanged, without `country`. No refit of any setting.
- **Calibration and threshold:** as the deployed system. The earliest 30% of the test window by booking date sets the isotonic calibrator and the 80%-recall threshold, and the other 70% evaluates.
- **Reported per cutoff:** average precision and ROC-AUC on the whole test window, and on the evaluation 70% the Brier score and ECE of the calibrated probability, the flagged share, recall and precision at the threshold, and the cancellation rate of the calibration and evaluation parts.
- **What counts as a finding:** the README states the range of each metric across the four cutoffs and whether the single-split values (average precision 0.6224, ECE 0.025, recall 66.3%) sit inside it. Recall below 80% at the evaluation window in a cutoff is reported as such. Nothing is retuned after the run.
- **Not claimed:** independence between cutoffs. Their training windows nest and their test windows overlap for the three earlier ones, so four numbers are not four independent draws.

### Result of the rolling backtest (same day)

`scripts/rolling_backtest.py` ran once as designed (`outputs/rolling_backtest.csv`). Average precision 0.571 to 0.735, ROC-AUC 0.670 to 0.788, ECE 0.026 to 0.060, recall at the threshold 66.0% to 90.3%, precision 44.5% to 48.5%. The recall target was met at two of four cutoffs (2016-07-12 and 2016-10-12) and missed at the other two, the last cutoff being the lowest. The single-split values are inside every range except ECE, where 0.025 is just below the smallest. The README states the ranges and says the earlier reading of the recall shortfall as a general property of the rule is not supported. Four cutoffs overlap and are not independent draws.


## Replay of the stated monitoring action, design fixed before it was run — 2026-11-14

Written before the rule below was scored. The README says the action on an alert is to re-set the threshold on recent matured outcomes and not to retrain. This replay tests whether that action would have recovered recall on the 27 evaluation weeks.

- **Trigger.** At the end of a week, if the rolling PSI status (trailing 2,000 scores, as `scripts/monitoring_replay.py`) is `alert` in that week and in the week before. No re-set within four weeks of the previous one.
- **Action.** The new threshold is the highest risk score reaching 80% recall among the latest 2,000 bookings, by booking date, whose scheduled stay had ended before the week's end. Only out-of-sample holdout scores are used. The pool needs at least 300 cancellations, otherwise the re-set is skipped and reported as skipped. The model, the calibrator and every other setting stay as deployed.
- **Maturity.** An outcome counts as known once the scheduled departure date has passed, as the README says outcomes arrive. A booking's cancellation date is not used, because cancellations are visible earlier than completed stays and that would bias the pool.
- **Compared.** The same weeks with the deployed threshold and with the rule: realised recall and precision by week, the mean of each over the weeks, the weeks below 80% recall, the flagged share, the number of re-sets and skips and the thresholds chosen.
- **What counts as a finding:** the README states the realised recall with and without the rule over the same 27 weeks. If the rule does not raise recall, that is reported as the result. Nothing is retuned after the run.
- **Limits:** one replay on one holdout, with a threshold chosen from the same period it then serves. The pool lags by the length of stay, so the rule acts later than the alert.

### Result of the action replay (same day)

`scripts/action_replay.py` ran once as designed (`outputs/monitoring/action_replay.json`). The rule re-set the threshold 6 times and skipped 2. Mean weekly recall 65.4% with the deployed threshold, 86.7% with the rule. Weeks below 80%: 27 against 7. Mean flagged share 44.2% against 72.9% and mean precision 42.0% against 34.4%. The recovery comes from flagging many more bookings. The first re-set (2017-02-27, threshold 0.211) put recall above 90% for six weeks because the pool of ended stays is short stays. The README reports the recovery and its cost. No setting was changed after the run. Leakage tests: the re-set threshold cannot depend on the outcome or the score of a booking whose stay has not ended.


## Wider lint rules — 2026-11-14

- Ruff now selects the rule families E, F, I, UP, B and SIM, with line length (E501) and zip without strict (B905) ignored, in `pyproject.toml`. The Makefile and the evidence workflow use that configuration. Before this the check selected syntax errors and unused names only.
- 19 import-sorting findings were fixed automatically. No behaviour changed. The monitoring replay was rerun after the change and its committed outputs are byte-identical, and the 43 tests and the evidence check pass.
