# Measuring retention uplift: experiment design

The decision app treats intervention success as an **assumption** because the
public dataset contains no retention intervention. This note describes how a
hotel would replace that assumption with a measured effect before scaling the
policy.

## Design

| Element | Choice |
|---|---|
| Population | Bookings whose risk score clears the review threshold (0.372, set for 80% recall) |
| Randomisation | Booking-level, 50/50 into contact vs. no contact, assigned at scoring time |
| Primary metric | Cancellation rate per arm, measured once the scheduled stay date has passed |
| Guardrails | Contact cost per booking, ADR of retained bookings, complaint/opt-out rate |
| Analysis | Two-sided two-proportion z-test, α = 0.05, power = 0.80; report the uplift with its confidence interval |
| Decision rule | Scale only if the lower confidence bound of the uplift still gives positive expected value at the observed contact cost |

Randomising **within** the flagged population is deliberate: the question is
whether contacting high-risk bookings changes their outcome, not whether
high-risk bookings cancel more (the model already ranks that).

## How large must the test be?

On the later-period holdout, 11,238 of 23,989 bookings clear the 0.372
risk-score threshold and **49.31%** of them cancelled (the holdout precision). That is the
control-arm baseline. Over the 33 weeks of holdout booking dates this is about
**341 flagged bookings per week** across both hotels.

Sample sizes from `experiment_sample_size()` in
[`decision.py`](../src/hotel_cancellation/decision.py):

| Relative reduction to detect | Per arm | Total | Approx. weeks of flagged volume |
|---:|---:|---:|---:|
| 30% (49.3% → 34.5%) | 174 | 348 | 1 |
| 20% (49.3% → 39.4%) | 398 | 796 | 2.3 |
| 15% (49.3% → 41.9%) | 711 | 1,422 | 4.2 |
| 10% (49.3% → 44.4%) | 1,607 | 3,214 | 9.4 |
| 5% (49.3% → 46.8%) | 6,447 | 12,894 | 38 |

```python
from hotel_cancellation.decision import experiment_sample_size

experiment_sample_size(baseline_cancellation_rate=0.4931, relative_reduction=0.10)
# 1607 bookings per arm
```

A realistic retention contact is more likely to move cancellations by single
digits than by 30%, so a hotel should plan for a test of roughly two months
rather than a one-week pilot. Outcomes also arrive only after the stay date,
so the readout lags enrolment by the typical lead time.

## Threats to validity

- **Interference:** contacted and uncontacted guests in the same group booking
  must be randomised together.
- **Seasonality:** run the test over full weeks and avoid stopping early when
  the interim result looks good (no peeking without a sequential design).
- **Threshold drift:** keep the scoring threshold fixed during the test, or the
  baseline rate changes under the experiment.
- **Displacement:** a retained booking may displace a later walk-in; measure
  revenue, not only the cancellation rate.

These figures use the historical holdout as a planning baseline. They are not
a measured effect, and nothing here has been run with real guests.
