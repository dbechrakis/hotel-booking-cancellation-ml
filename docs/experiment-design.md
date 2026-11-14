# Measuring retention uplift: experiment design

The decision app treats intervention success as an **assumption** because the
public dataset contains no retention intervention. This note describes how a
hotel would replace that assumption with a measured effect before scaling the
policy.

## Design

| Element | Choice |
|---|---|
| Population | Bookings whose risk score clears the review threshold (0.469, set for 80% recall) |
| Randomisation | Booking-level, 50/50 into contact vs. no contact, assigned at scoring time |
| Primary metric | Cancellation rate per arm, measured once the scheduled stay date has passed |
| Guardrails | Contact cost per booking, ADR of retained bookings, complaint/opt-out rate |
| Analysis | Two-sided two-proportion z-test, α = 0.05, power = 0.80; report the uplift with its confidence interval |
| Decision rule | Scale only if the lower confidence bound of the uplift still gives positive expected value at the observed contact cost |

Randomising **within** the flagged population is deliberate: the question is
whether contacting high-risk bookings changes their outcome, not whether
high-risk bookings cancel more (the model already ranks that).

## How large must the test be?

The baseline should come from the regime the test would run in. On the 16,835
later bookings (after the window where the threshold and calibrator were set),
7,589 clear the 0.469 risk-score threshold and **42.89%** of them cancelled. That
is the control-arm baseline. Over the 29 weeks of those booking dates this is
about **260 flagged bookings per week** across both hotels.

An earlier version of this page used all 23,989 holdout bookings (47.51%, 343 a
week). That pooled in the threshold window, whose cancellation rate was higher
(37% against 29%) because non-refundable bookings later almost disappeared (see
the README's monitoring section). It understated the test's length by about a quarter.

Sample sizes from `experiment_sample_size()` in
[`decision.py`](../src/hotel_cancellation/decision.py):

| Relative reduction to detect | Per arm | Total | Approx. weeks of flagged volume |
|---:|---:|---:|---:|
| 30% (42.9% → 30.0%) | 219 | 438 | 1.7 |
| 20% (42.9% → 34.3%) | 505 | 1,010 | 3.9 |
| 15% (42.9% → 36.5%) | 907 | 1,814 | 7.0 |
| 10% (42.9% → 38.6%) | 2,060 | 4,120 | 15.8 |
| 5% (42.9% → 40.7%) | 8,304 | 16,608 | 63.8 |

```python
from hotel_cancellation.decision import experiment_sample_size

experiment_sample_size(baseline_cancellation_rate=0.4289, relative_reduction=0.10)
# 2060 bookings per arm
```

A realistic retention contact is more likely to move cancellations by single
digits than by 30%, so a hotel should plan for a test of three to four months
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
