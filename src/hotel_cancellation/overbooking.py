"""Count forecasts for the bookings on hand and the room limit they imply."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

COST_RATIOS = (1, 2, 3)
DEMAND_SHARES = (0.25, 0.5, 1.0)


def poisson_binomial(probabilities) -> np.ndarray:
    """Exact distribution of the number of successes among independent yes or no outcomes, by convolution."""
    pmf = np.array([1.0])
    for p in np.asarray(probabilities, dtype=float):
        if not 0 <= p <= 1:
            raise ValueError("Probabilities must be between 0 and 1")
        pmf = np.concatenate([pmf * (1 - p), [0.0]]) + np.concatenate([[0.0], pmf * p])
    return pmf


def mean_and_variance(distribution: np.ndarray) -> tuple[float, float]:
    support = np.arange(len(distribution))
    mean = float(support @ distribution)
    return mean, float(((support - mean) ** 2) @ distribution)


def widen(distribution: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the variance by ``factor`` by convolving with a discretised normal of the missing variance.

    Mass pushed outside 0..n is folded into the end points, so the mean and variance hold away from the ends."""
    if factor < 1:
        raise ValueError("The factor cannot be below 1")
    distribution = np.asarray(distribution, dtype=float)
    _, variance = mean_and_variance(distribution)
    missing = (factor - 1) * variance
    if missing <= 0:
        return distribution.copy()
    radius = int(np.ceil(6 * np.sqrt(missing))) + 1
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-(offsets**2) / (2 * missing))
    weights /= weights.sum()
    n = len(distribution) - 1
    out = np.zeros(n + 1)
    for shift, weight in zip(offsets, weights, strict=True):
        target = np.clip(np.arange(n + 1) + shift, 0, n)
        np.add.at(out, target, distribution * weight)
    return out


def horizon_calibrator(scores, failed) -> IsotonicRegression:
    """Map the risk score to the chance that a booking alive at the horizon later fails to show. Increasing in the score."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(scores, failed)


def segment_rates(frame: pd.DataFrame, prior_weight: float = 50.0) -> dict:
    """Failure rate by hotel, deposit type and market segment, each cell shrunk toward its hotel rate by n over n plus the weight."""
    hotel_rate = frame.groupby("hotel")["failed"].mean()
    cells = frame.groupby(["hotel", "deposit_type", "market_segment"])["failed"].agg(["sum", "count"])
    rates = {}
    for (hotel, deposit, segment), row in cells.iterrows():
        weight = row["count"] / (row["count"] + prior_weight)
        rates[(hotel, deposit, segment)] = weight * row["sum"] / row["count"] + (1 - weight) * hotel_rate[hotel]
    return {"hotel": hotel_rate.to_dict(), "cells": rates}


def segment_failure_probability(rates: dict, frame: pd.DataFrame) -> np.ndarray:
    keys = zip(frame["hotel"], frame["deposit_type"], frame["market_segment"], strict=True)
    return np.array([rates["cells"].get(key, rates["hotel"][key[0]]) for key in keys])


def crps_discrete(distribution, observed: int) -> float:
    """Continuous ranked probability score of an integer-valued forecast, in the unit of the count."""
    distribution = np.asarray(distribution, dtype=float)
    size = max(len(distribution), int(observed) + 1)
    padded = np.zeros(size)
    padded[: len(distribution)] = distribution
    cdf = np.cumsum(padded)
    step = (np.arange(size) >= observed).astype(float)
    return float(np.sum((cdf - step) ** 2))


def interval(distribution, level: float) -> tuple[int, int]:
    """Central interval of an integer-valued forecast. Both end points are included."""
    if not 0 < level < 1:
        raise ValueError("The level must be between 0 and 1")
    cdf = np.cumsum(np.asarray(distribution, dtype=float))
    tail = (1 - level) / 2
    low = int(np.searchsorted(cdf, tail - 1e-12, side="left"))
    high = int(np.searchsorted(cdf, 1 - tail - 1e-12, side="left"))
    return low, min(high, len(cdf) - 1)


def tail_probability(distribution: np.ndarray, threshold: int) -> float:
    """P(S >= threshold)."""
    if threshold <= 0:
        return 1.0
    if threshold >= len(distribution):
        return 0.0
    return float(distribution[threshold:].sum())


def extra_rooms(
    distribution_of_shows,
    show_probability: float,
    capacity: int,
    cost_ratio: float,
    max_extra: int = 40,
) -> int:
    """Largest number of rooms to sell above capacity.

    Room k is accepted while the chance that the guests arriving exceed capacity stays at or below 1 over 1 plus the cost ratio.
    ``distribution_of_shows`` already includes the pickup still to come."""
    if cost_ratio <= 0:
        raise ValueError("The cost ratio must be positive")
    critical = 1.0 / (1.0 + cost_ratio)
    base = np.asarray(distribution_of_shows, dtype=float)
    accepted = 0
    for k in range(1, max_extra + 1):
        extra = poisson_binomial([show_probability] * k)
        total = np.convolve(base, extra)
        if tail_probability(total, capacity + 1) > critical + 1e-12:
            break
        accepted = k
    return accepted


def scenario_value(extra: int, demand_share: float, show_rate: float, free_rooms: float, cost_ratio: float) -> dict:
    """Value of the extra rooms in units of one room night, under a stated share of them being booked."""
    if not 0 <= demand_share <= 1 or not 0 <= show_rate <= 1:
        raise ValueError("Shares must be between 0 and 1")
    arriving = demand_share * extra * show_rate
    free = max(free_rooms, 0.0)
    filled = min(arriving, free)
    walked = max(arriving - free, 0.0)
    return {"extra_shows": arriving, "gain": filled, "walks": walked, "net": filled - cost_ratio * walked}


def refit_date(decision_date) -> pd.Timestamp:
    """First day of the calendar month of a decision date. Methods are refitted on that day."""
    return pd.Timestamp(decision_date).to_period("M").start_time


def eligible_pool(arrival: pd.Series, as_of) -> pd.Series:
    """Bookings whose outcome is known at ``as_of``: those that arrived before it."""
    return arrival < pd.Timestamp(as_of)


def overdispersion_factor(squared_errors, variances, minimum_nights: int = 7) -> float:
    """Ratio of observed squared error to predicted variance over past nights, never below 1."""
    squared_errors = np.asarray(squared_errors, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if len(squared_errors) < minimum_nights or variances.sum() <= 0:
        return 1.0
    return float(max(1.0, squared_errors.sum() / variances.sum()))
