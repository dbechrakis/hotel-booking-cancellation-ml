# Probability calibration

The deployed logistic model is trained with `class_weight="balanced"` so that the 0.50
threshold favours recall. The side effect is that its output overstates cancellation
rates. On later bookings the raw score averages **51%** while **29%** actually cancelled.
Ranking and the threshold are unaffected. Expected intervention value, however, multiplies
a probability by money, and the risk bands describe likelihood, so both need a calibrated
number.

The system now serves two numbers per booking:

| Output | Used for | Source |
|---|---|---|
| `risk_score` | ranking, the 0.50 review threshold, the threshold simulator, all classification metrics | logistic pipeline (unchanged role) |
| `cancellation_probability` | risk bands, expected intervention value | isotonic regression applied to the risk score |

Isotonic regression is monotone, so it never reorders bookings. Its step function does
create ties, which is why ranking stays on the raw score.

## Protocol

1. The logistic pipeline is trained on matured bookings before 2017-01-12, exactly as before.
2. The isotonic calibrator is fitted on the **earliest 30% of later bookings**
   (7,154 bookings made before 2017-02-08).
3. Calibration is evaluated on the **remaining 16,835 later bookings**, which neither
   the model nor the calibrator saw.

This mirrors production practice: recalibrate a deployed score on its most recent matured
outcomes. One caveat: in live operation those outcomes would only be known once the
bookings' stays had passed, so a real calibrator would lag by the typical lead time.

## Results on the 16,835 evaluation bookings

| | Brier | Log loss | ECE (10 bins) | Mean predicted | Observed |
|---|---:|---:|---:|---:|---:|
| Raw risk score | 0.229 | 0.653 | 0.221 | 51.2% | 29.2% |
| **Isotonic probability** | **0.174** | **0.519** | **0.013** | **30.1%** | 29.2% |

![Reliability diagram](../outputs/calibration_reliability.png)

Bin-level values are in [`outputs/calibration_reliability.csv`](../outputs/calibration_reliability.csv).
Platt scaling on the same split gave a Brier score of 0.175, slightly worse; isotonic was
kept.

**What changes for a user:** for the app's default booking, the risk score is 0.88 but the
calibrated probability is 60%. With the default assumptions (margin 100, cost 5, success 25%),
the expected intervention value falls from about 17 to 10. The old figure overstated the
value of every contact.

## What did not work, and why

Fitting the calibrator inside the training period, on an inner chronological validation
window, made calibration *worse* on later bookings (mean predicted 49.5% vs 31.5% observed).
The cause is a feature-contract issue: `arrival_date_year` is a numeric feature, so a
model fitted on an earlier window extrapolates the year term differently from the final
model. The two models' scores did not share a scale. Forward recalibration on the final
model's own scores avoids the mismatch. A better long-term fix is to drop or bucket
`arrival_date_year` so scores stop extrapolating beyond the training years; that changes
the model and needs its own revalidation.
