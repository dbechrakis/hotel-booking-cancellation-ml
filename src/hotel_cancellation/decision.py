"""Decision rules and threshold economics layered above model probabilities."""

import numpy as np
import pandas as pd


def risk_band(probability: float) -> str:
    """Map a probability to a transparent portfolio risk band."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    if probability < 0.30:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.70:
        return "High"
    return "Very high"


def expected_intervention_value(
    probability: float,
    recoverable_margin: float,
    intervention_cost: float,
    intervention_success_rate: float,
) -> float:
    """Estimate expected value under explicit, user-controlled assumptions."""
    if recoverable_margin < 0 or intervention_cost < 0:
        raise ValueError("Economic inputs must be non-negative")
    if not 0 <= intervention_success_rate <= 1:
        raise ValueError("Success rate must be between 0 and 1")
    return probability * intervention_success_rate * recoverable_margin - intervention_cost


def recommendation(probability: float, policy_threshold: float, expected_value: float) -> str:
    """Recommend review only when risk and assumed economics both support action."""
    if probability >= policy_threshold and expected_value > 0:
        return "Prioritize for retention review"
    if probability >= policy_threshold:
        return "High risk, but intervention economics are negative"
    if expected_value > 0:
        return "Positive expected value, but below the current policy threshold"
    return "No proactive intervention under current assumptions"


def threshold_table(
    y_true,
    probabilities,
    recoverable_margin: float,
    intervention_cost: float,
    intervention_success_rate: float,
    thresholds=None,
) -> pd.DataFrame:
    """Evaluate threshold trade-offs and assumed intervention value on a holdout."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("Targets and probabilities must be non-empty and aligned")
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.05)
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        interventions = int(predicted.sum())
        precision = tp / interventions if interventions else 0.0
        recall = tp / int((y == 1).sum()) if (y == 1).any() else 0.0
        net_value = (
            tp * intervention_success_rate * recoverable_margin
            - interventions * intervention_cost
        )
        rows.append(
            {
                "threshold": float(threshold),
                "interventions": interventions,
                "true_positives": tp,
                "false_positives": fp,
                "missed_cancellations": fn,
                "precision": precision,
                "recall": recall,
                "assumed_net_value": net_value,
            }
        )
    return pd.DataFrame(rows)
