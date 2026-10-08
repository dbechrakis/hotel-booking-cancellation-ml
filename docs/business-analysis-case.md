# Business analysis case: cancellation review

This is a **portfolio design exercise** based on the public Hotel Booking Demand data and the implemented application. It is not a record of hotel stakeholder interviews, a commissioned requirement, client UAT, or a production rollout. The acceptance scenarios below specify what a prospective reviewer could check; their presence does not claim that a hotel has signed them off.

## Decision and users

**Proposed user:** a reservations or retention analyst with limited capacity for proactive review. The user wants to identify bookings worth examining before arrival and compare how a contact threshold affects workload and historical cancellation coverage.

**Decision:** place a booking in a *review queue* when its estimated cancellation probability reaches the chosen threshold **and** the assumed expected intervention value is positive. The application recommends review; it does not contact a guest or choose an offer. The threshold simulator compares possible queue sizes on one saved later-period holdout.

**Success criterion for a future pilot:** a controlled intervention study would have to measure incremental retained margin net of contact costs, alongside workload and adverse effects. This dataset cannot establish that criterion because it contains no randomized retention action.

## Requirements trace

| ID | User need | Current behavior | Acceptance criterion and evidence |
|---|---|---|---|
| BA-01 | Assess a booking with information available for a prospective decision. | The form collects the explicit 21-feature contract and the exported pipeline scores it. | A submitted form returns a probability between 0 and 1; outcome-time fields such as final status and assigned room are absent from the feature contract. See [`contracts.py`](../src/hotel_cancellation/contracts.py), [`streamlit_app.py`](../app/streamlit_app.py), and [validation](../VALIDATION.md). |
| BA-02 | Understand the score and proposed action. | The scoring view displays a risk band, largest signed logistic contributions, and one of four recommendation messages. | Review is recommended only when probability is at least the policy threshold **and** expected value is greater than zero. Contributions are labeled associations, not causes. See [`decision.py`](../src/hotel_cancellation/decision.py) and [`streamlit_app.py`](../app/streamlit_app.py). |
| BA-03 | Change the economic assumptions transparently. | The user can set recoverable margin, contact cost, assumed success rate, and policy threshold. | Per-booking value equals `probability × success rate × recoverable margin − cost`. Increasing cost by a fixed amount reduces that value by the same amount for the same booking. See [`decision.py`](../src/hotel_cancellation/decision.py). |
| BA-04 | Compare workload with historical coverage. | The simulator shows contacts, precision, cancellation recall, assumed net value, and a table with missed cancellations for thresholds from 0.10 to 0.90. | Raising the threshold cannot increase the number contacted or cancellation recall on the fixed holdout. Net value uses observed true positives only as a scenario input, not observed prevented cancellations. See [`decision.py`](../src/hotel_cancellation/decision.py). |
| BA-05 | Inspect the basis and limits before use. | The Model evidence tab shows row counts, holdout metrics, split cutoff, and limitations. | The user can distinguish the 23,989-row later holdout from the training set and can find the recorded validation scope. See [validation](../VALIDATION.md). |

## Acceptance walkthrough for a prospective reviewer

| Scenario | Steps | Expected observation |
|---|---|---|
| Score a booking | Open **Score a booking**, leave the example inputs at their defaults, and select **Score booking**. | Probability, risk band, assumed expected value, a recommendation, and signed contributions appear. The precise score depends on the committed model and these inputs; it is not a promised business outcome. |
| Negative economics at high risk | Score the same booking with an assumed success rate of **0** and a positive intervention cost. If its risk meets the selected threshold, compare the message. | The expected value is negative and the message says **“High risk, but intervention economics are negative”**; no review priority is suggested. If risk falls below the threshold, the message instead says **“No proactive intervention under current assumptions.”** |
| Threshold trade-off | In **Threshold simulator**, compare the table rows at thresholds **0.10** and **0.90** without changing the economic assumptions. | Contacts and recall at 0.90 do not exceed those at 0.10. The table also exposes missed cancellations and the assumed net-value calculation. |
| Evidence boundary | Open **Model evidence** and follow the repository validation link. | The historical split, metrics and caveats are visible. The simulator's reused holdout and assumed intervention success are not presented as an independent test or measured uplift. |

The rules in `decision.py` have automated unit coverage in [`tests/test_decision.py`](../tests/test_decision.py). The walkthrough is a proposed human acceptance check, not a claim of completed stakeholder UAT.

## Open questions before operational use

1. Which booking fields are actually available and lawful to use at review time, and how often do they change?
2. What is the capacity and true marginal cost of contacting a guest? Who owns the final action?
3. Are probabilities sufficiently calibrated for an economic threshold across hotels and time periods?
4. What controlled experiment could estimate incremental retention, margin, guest experience, and fairness effects?
