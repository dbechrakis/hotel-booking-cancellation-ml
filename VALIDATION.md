# Validation record

**Scope of the latest review.** Rerun in the review of 2026-11-14: training of the deployed artifacts without `country`, the country comparison, the rolling backtest, the action replay, the monitoring replay, the unit tests, the evidence check and lint. Recorded history and not rerun in that review: the notebook comparison and the tracked MLflow experiments, which keep `country`, and the Hugging Face deployment. Command: `make check`, then `PYTHONPATH=src python scripts/train_decision_artifacts.py`.

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

## Second review of the country removal — 2026-11-14

- A second reader checked the README against the new artifacts. One statement was off: non-refundable bookings were said to score 0.99 or more. Under the model without `country` they score 0.97 or more (85% of them in the threshold window are at 0.99 or more). The README is corrected. Their shares, 13.1% of the threshold window and 1.4% afterwards, are unchanged.
- The screenshot in the README was taken before the removal and is now labelled as such. It has not been retaken.

## Overbooking study, design fixed before it was run (2026-11-15)

Written after the counts of the gate and before any forecast was scored. The counts are in `outputs/overbooking/counts.json`.

**Question.** First, for a night and a number of days before it, how well does each method predict how many of the bookings on hand will show up. Second, given that forecast, what room limit does a cost rule recommend, and what would it have gained or cost under stated demand scenarios. The first part is scored exactly. The second part is scenario arithmetic and is labelled so.

**Definitions.**

- Booking date is arrival date minus `lead_time`, as `data.py` computes it. Departure is arrival plus nights. A booking covers night d when arrival is on or before d and d is before departure. Bookings with zero nights are left out of the room ledger.
- The decision date for night d at horizon h is t = d minus h days.
- On hand at (d, h): bookings that cover d, with booking date on or before t, not cancelled by t. A cancelled booking's `reservation_status_date` is its cancellation date. A no-show counts as not showing up and is on hand until the night.
- A booking shows when `is_canceled` is 0.
- Pickup at (d, h): bookings that cover d with booking date after t and that show.
- Capacity C, per hotel: the largest number of occupied rooms on any night before 2017-01-12. The file has no capacity field, so this is an inference. The recomputed values are 226 for the City Hotel and 187 for the Resort Hotel, equal to the feasibility counts. The 95th percentile of occupancy before the cutoff is 224 and 183.
- Horizons: 30, 14, 7 and 3 days.

**Which bookings can be scored honestly.** The deployed model trained on bookings whose stay ended before 2017-01-12. Every booking that was not in that training set, including the rows the split calls purged, is scored with the frozen artifact. Nothing is retrained.

**Methods.** Each gives the probability that a booking alive at t shows up.

| Method | Probability of showing |
|---|---|
| `flat` | One minus the share of on-hand bookings at horizon h that later failed to show, per hotel, from past nights only. |
| `segment` | The same by hotel, `deposit_type` and `market_segment`, each cell shrunk toward the hotel rate with weight n over n plus 50. |
| `at_booking` | One minus the deployed calibrated probability, unchanged. It is expected to be badly wrong because it is made at booking time. It is included to show the size of the error. |
| `model` | An isotonic calibrator of "later failed to show" on the deployed `risk_score`, fitted per horizon on past bookings that were alive at that horizon. |
| `model_od` | `model`, with the count distribution widened by an overdispersion factor estimated from past nights. |

A method used at decision date t may learn only from bookings that arrived before t. Methods are refitted on the first day of each calendar month from all eligible past bookings. The calibrator and the baselines learn only from bookings the frozen model did not train on.

**Evaluation nights.** From 2017-04-01 to 2017-08-31, both hotels, four horizons. Gate, counts only: at the first decision date of each horizon the pool must hold at least 1,000 bookings alive at that horizon with at least 50 that later failed to show, per hotel. If a horizon fails, the start moves two weeks later for all horizons, at most twice.

**Gate result.** The first start, 2017-04-01, failed. At the first decision date of the 30-day horizon, 2017-03-02, the Resort Hotel pool held 739 alive bookings with 122 failures, below 1,000. The start moved two weeks to **2017-04-15**. At that start every horizon and hotel passes, with 1,024 alive bookings for the Resort Hotel at 30 days as the smallest pool. There are **139 evaluation nights** per hotel, from 2017-04-15 to 2017-08-31.

**The count forecast.** The number of on-hand bookings that show is a sum of independent yes or no outcomes with different probabilities. Its distribution, the Poisson binomial, is computed exactly by convolution. Bookings are not independent, since a group cancels together, so the exact distribution is expected to be too narrow. `model_od` widens it by mixing with a discretised normal of the missing variance. Scores per method and horizon, pooled over nights and hotels: mean error in rooms (predicted mean minus observed shows), continuous ranked probability score in rooms, and coverage of the central 80% and 95% intervals. Intervals for the scores come from a block bootstrap over calendar weeks, because adjacent nights share most of their bookings.

**The room limit.** The hotel earns r for a room filled and pays w for a guest it cannot house, with r equal to 1. Accept extra room k while P(S_k at least C plus 1) is at most r over r plus w. S_k is the shows from the bookings on hand, plus k more bookings that each show with the method's average probability, plus the pickup still to come. The pickup forecast is the empirical distribution of pickup at this horizon for this hotel and weekday, from past nights only, convolved in. Cost ratios w over r are 1, 2 and 3, and the middle one is the headline. No currency figure is claimed.

**Scenario value.** A hotel that does not overbook records no turned-away guests, so the value is computed under a stated share of the extra rooms that would in fact have been booked. A constrained night is one where the bookings on hand at the horizon are at least 90% of capacity. Extra bookings taken are the demand share times k on constrained nights and 0 elsewhere, for demand shares of 0.25, 0.5 and 1. Their show rate is the realised show rate of that night's bookings made after the decision date, or the hotel's rate when the night has fewer than 10 such bookings. Rooms free on the night are capacity minus the rooms actually occupied, and not below zero. Gain is extra shows up to the free rooms. Walks are extra shows beyond them. Net value is gain minus the cost ratio times walks. Tables report it per hotel per 100 nights next to "no overbooking", which is zero by definition, and each carries the line: scenario arithmetic under an assumed demand share, not a measured outcome.

**Implementation rules, fixed before the run.**

- The pool at a refit date R is every booking the frozen model did not train on that arrived before R and was alive at its own decision date for horizon h. The first refit is 2017-02-01. Forecasts are produced for every night from 2017-02-01 plus h days to 2017-08-31. Only nights from the evaluation start are scored.
- The overdispersion factor is updated daily. At decision date t it is, per hotel and horizon, the sum of squared errors over the sum of predicted variances of `model` over nights before t that have a forecast. It is 1 with fewer than 7 such nights and never below 1.
- The pickup distribution uses the 16 most recent same-weekday nights of that hotel before t, from the whole history.
- The show probability of an extra booking is the mean of the method's probabilities over the bookings on hand. The rule is searched up to 40 extra rooms.
- Central intervals come from the discrete distribution and include both end points. The bootstrap has 1,000 resamples of calendar weeks with seed 42.
- The sensitivity line reports the scenario value with capacity minus 5 and plus 5, at the middle cost ratio and a demand share of 0.5.

**Expectations.**

- O1. `at_booking` over-predicts the rooms lost at every horizon. Its mean error is negative (fewer shows predicted than realised) by more than 5 rooms per night at the 7-day horizon for the City Hotel.
- O2. `model` has a lower continuous ranked probability score than `flat` at the 30-day and 14-day horizons. No prediction at 7 and 3 days, where little is left to predict.
- O3. `model` has a lower score than `segment` at the 30-day horizon by more than the bootstrap standard error of the difference. If the difference is within one standard error, the reading is "the booking-level model adds nothing over three segment rates", and that sentence goes in the README.
- O4. The exact count distribution is too narrow. The 80% interval of `model` covers fewer than 70% of nights at the 30-day horizon.
- O5. `model_od` covers between 70% and 90% of nights with its 80% interval at every horizon.
- O6. At a cost ratio of 2 and the 30-day horizon, the rule under `model_od` recommends at least one extra room on more than half of the constrained nights.
- O7. No prediction on the sign of the net value. Reported for all nine combinations of cost ratio and demand share.

**What counts.** Each expectation is graded by its own rule from `outputs/overbooking/summary.json`. O1 is graded for the City Hotel at the 7-day horizon on the mean error, and it also requires a negative mean error at every horizon for both hotels. O2 to O5 are graded on scores pooled over both hotels and all evaluation nights. O6 pools both hotels. The methods, horizons, evaluation window, capacity rule, cost ratios and demand shares are not changed after the first run.

**Not claimed.** That the capacity is the hotels' true room count. That demand existed for the extra rooms. That a walk costs a fixed multiple of a room night. Any effect on guests, reviews or repeat business. Independence between nights.

## Overbooking study, result (2026-11-15)
Run once with `scripts/overbooking_backtest.py` on the design fixed above (design commit `59ca1c6`). Nothing was changed after the run. An earlier launch of the same script stopped on a date-type error in the score table, before any score was written or read, and was fixed and rerun. Numbers come from `outputs/overbooking/summary.json` and the CSV files next to it, and `ci/verify_evidence.py` recomputes the ones the README quotes. The scenario figures are scenario arithmetic under an assumed demand share, not a measured outcome. Failed expectations come first.
- O2. Failed. At the 30-day horizon `model` scores 9.23 rooms against 7.35 for `flat`. At 14 days it scores 6.04 against 5.39. A lower score is better, so the booking-level model is worse than one flat rate by horizon.
- O3. Failed. At the 30-day horizon `segment` minus `model` is -3.13 rooms, with a bootstrap standard error of 0.37 (interval -3.89 to -2.46). The model is worse than three segment rates by more than eight standard errors, so the difference is not within one standard error. The booking-level model adds nothing over three segment rates here, and does worse.
- O5. Failed. `model_od` covers 95.3%, 88.1%, 93.5% and 96.4% of nights with its 80% interval at 30, 14, 7 and 3 days. Only the 14-day value is inside 70% to 90%. The widened distribution is too wide. The factor is estimated from squared errors that include the bias, and every method under-predicts shows, so the factor absorbs the bias as variance.
- O6. Failed. At a cost ratio of 2 and 30 days, `model_od` recommends an extra room on 12.8% of the 117 constrained hotel nights, not more than half.
- O1. Passed. The mean error of `at_booking` is negative at every horizon for both hotels. For the City Hotel at 7 days it is -69.7 rooms per night. It predicts about 72 too few shows at 3 days.
- O4. Passed. The 80% interval of `model` covers 21.2% of nights at 30 days, far below 70%. The exact distribution is far too narrow.
- O7. Not graded. The net value is reported for all nine combinations of cost ratio and demand share in `scenarios.csv`. Across the 20 method and horizon cells, the number with a negative net value (both hotels, no capacity shift) is: cost ratio 1, demand shares 0.25, 0.5 and 1: 0, 4 and 4. Cost ratio 2: 2, 4 and 4. Cost ratio 3: 4, 4 and 5. The negative cells are mostly `at_booking`, which sells far too many rooms.

**What else the run showed.** Every method under-predicts shows. The bias of `flat` is -8.5 rooms per night at 30 days. A cause was not tested. One reading is that failure rates learned in January to March were higher than in April to August. The first start of the evaluation window, 2017-04-01, failed the gate and the window started on 2017-04-15, with 139 nights per hotel.

**Limits.** Capacity is the highest occupancy seen before 2017-01-12, not a room count. The data records no turned-away demand, so the value depends on the assumed demand share. About 139 nights for each of two hotels in one spring and summer, and adjacent nights share bookings. The frozen model was built to rank bookings at booking time and is reused with a horizon calibrator. Room types are ignored. The frozen artifact scored some bookings with zero guests, which its API contract would reject. They were scored directly through the pipeline and the calibrator, which were not modified.
