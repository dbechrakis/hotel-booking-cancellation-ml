"""Score-distribution monitoring for the deployed model.

Outcomes arrive only after a booking's stay, so a live service cannot measure recall when
it scores. It can watch what it is fed: whether the risk-score distribution, the flagged
share and the share of unseen categories have moved away from the window the threshold and
calibrator were set on. A large shift is the early warning that recall may have moved too.
"""

from collections import deque
import threading

import numpy as np


# Conventional population-stability bands.
PSI_WARN = 0.10
PSI_ALERT = 0.25


def reference_profile(scores: np.ndarray, probabilities: np.ndarray, threshold: float, bins: int = 10) -> dict:
    """Decile edges and summary rates of the scores the threshold was chosen on."""
    edges = np.quantile(scores, np.linspace(0, 1, bins + 1))[1:-1]
    counts = np.bincount(np.searchsorted(edges, scores, side="right"), minlength=bins)
    return {
        "bin_edges": [float(edge) for edge in edges],
        "bin_shares": [float(count / len(scores)) for count in counts],
        "flagged_share": float(np.mean(scores >= threshold)),
        "mean_probability": float(np.mean(probabilities)),
        "rows": int(len(scores)),
    }


def population_stability_index(reference_shares, actual_scores, edges, floor: float = 1e-4) -> float:
    """PSI of new scores against reference bin shares (0 = identical distribution)."""
    expected = np.clip(np.asarray(reference_shares, dtype=float), floor, None)
    counts = np.bincount(np.searchsorted(np.asarray(edges), actual_scores, side="right"), minlength=len(expected))
    actual = np.clip(counts / max(1, len(actual_scores)), floor, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WARN:
        return "warn"
    return "ok"


class ScoreMonitor:
    """Thread-safe rolling window of recent scores and request statistics."""

    def __init__(self, reference: dict, window: int = 5000, min_rows: int = 200):
        self.reference = reference
        self.min_rows = min_rows
        self.scores: deque[float] = deque(maxlen=window)
        self.probabilities: deque[float] = deque(maxlen=window)
        self.flagged: deque[bool] = deque(maxlen=window)
        self.unseen: deque[bool] = deque(maxlen=window)
        self.latency_ms: deque[float] = deque(maxlen=1000)
        self.total = 0
        self.lock = threading.Lock()

    def record(self, scores, probabilities, flagged, unseen, latency_ms: float) -> None:
        with self.lock:
            self.scores.extend(float(s) for s in scores)
            self.probabilities.extend(float(p) for p in probabilities)
            self.flagged.extend(bool(f) for f in flagged)
            self.unseen.extend(bool(u) for u in unseen)
            self.latency_ms.append(float(latency_ms))
            self.total += len(scores)

    def summary(self) -> dict:
        with self.lock:
            scores = np.array(self.scores)
            probabilities = np.array(self.probabilities)
            flagged = np.array(self.flagged)
            unseen = np.array(self.unseen)
            latency = np.array(self.latency_ms)
        reference = self.reference
        result = {
            "bookings_scored_total": self.total,
            "window_rows": len(scores),
            "reference_rows": reference["rows"],
            "reference_flagged_share": reference["flagged_share"],
            "reference_mean_probability": reference["mean_probability"],
        }
        if len(scores) == 0:
            return {**result, "status": "no_traffic"}
        result.update(
            {
                "flagged_share": float(flagged.mean()),
                "mean_probability": float(probabilities.mean()),
                "unseen_category_share": float(unseen.mean()),
                "latency_ms_p50": float(np.percentile(latency, 50)),
                "latency_ms_p95": float(np.percentile(latency, 95)),
            }
        )
        if len(scores) < self.min_rows:
            return {**result, "status": "insufficient_data", "psi": None}
        psi = population_stability_index(reference["bin_shares"], scores, reference["bin_edges"])
        return {**result, "psi": psi, "status": drift_status(psi)}
