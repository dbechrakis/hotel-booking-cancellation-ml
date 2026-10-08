"""Calibration diagnostics for cancellation probabilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def reliability_table(outcomes, probabilities, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed cancellation rate in (roughly) equal-count probability bins.

    Bins are cut on the values, so tied predictions from the isotonic step function always
    share a bin and the result does not depend on row order; ties can merge bins.
    """
    frame = pd.DataFrame({"outcome": np.asarray(outcomes, dtype=float), "p": np.asarray(probabilities, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    table = frame.groupby("bin").agg(
        bookings=("outcome", "size"), mean_predicted=("p", "mean"), observed_rate=("outcome", "mean")
    )
    return table.reset_index(drop=True)


def expected_calibration_error(outcomes, probabilities, bins: int = 10) -> float:
    table = reliability_table(outcomes, probabilities, bins)
    weights = table["bookings"] / table["bookings"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def calibration_summary(outcomes, probabilities) -> dict:
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(outcomes, probabilities)),
        "log_loss": float(log_loss(outcomes, probabilities)),
        "ece": expected_calibration_error(outcomes, probabilities),
        "mean_predicted": float(probabilities.mean()),
        "observed_rate": float(outcomes.mean()),
    }
