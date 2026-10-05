# Measuring retention uplift: experiment design

The decision app treats intervention success as an **assumption** because the
public dataset contains no retention intervention. This note describes how a
hotel would replace that assumption with a measured effect before scaling the
policy.

## Design

| Element | Choice |
|---|---|
| Population | Bookings whose cancellation probability clears the policy threshold (0.50 by default) |
| Randomisation | Booking-level, 50/50 into contact vs. no contact, assigned at scoring time |
| Primary metric | Cancellation rate per arm, measured once the scheduled stay date has passed |
| Guardrails | Contact cost per booking, ADR of retained bookings, complaint/opt-out rate |
| Analysis | Two-sided two-proportion z-test, α = 0.05, power = 0.80; report the uplift with its confidence interval |
| Decision rule | Scale only if the lower confidence bound of the uplift still gives positive expected value at the observed contact cost |

Randomising **within** the flagged population is deliberate: the question is
whether contacting high-risk bookings changes their outcome, not whether
high-risk bookings cancel more (the model already ranks that).

## How large must the test be?

On the later-period holdout, 12,929 of 23,989 bookings clear the 0.50
threshold and **46.76%** of them cancelled (the holdout precision). That is the
control-arm baseline. Over the 33 weeks of holdout booking dates this is about
**392 flagged bookings per week** across both hotels.

Sample sizes from `experiment_sample_size()` in
[`decision.py`](../src/hotel_cancellation/decision.py):

| Relative reduction to detect | Per arm | Total | Approx. weeks of flagged volume |
|---:|---:|---:|---:|
| 30% (46.8% → 32.7%) | 190 | 380 | 1 |
| 20% (46.8% → 37.4%) | 437 | 874 | 2.2 |
| 15% (46.8% → 39.7%) | 782 | 1,564 | 4 |
| 10% (46.8% → 42.1%) | 1,772 | 3,544 | 9 |
| 5% (46.8% → 44.4%) | 7,122 | 14,244 | 36 |

```python
from hotel_cancellation.decision import experiment_sample_size

experiment_sample_size(baseline_cancellation_rate=0.4676, relative_reduction=0.10)
# 1772 bookings per arm
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
