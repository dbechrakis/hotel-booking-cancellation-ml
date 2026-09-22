"""Chronological split and feature preparation."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

from hotel_cancellation.contracts import FEATURES, TARGET


@dataclass(frozen=True)
class TemporalSplit:
    cutoff: pd.Timestamp
    train_mask: pd.Series
    test_mask: pd.Series
    purged_mask: pd.Series
    booking_date: pd.Series


def source_sha256(path: Path) -> str:
    """Return the SHA-256 fingerprint of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chronological_split(frame: pd.DataFrame, quantile: float = 0.8) -> TemporalSplit:
    """Create the notebook's later-booking holdout with matured training labels."""
    arrival = pd.to_datetime(
        frame["arrival_date_year"].astype(str)
        + "-"
        + frame["arrival_date_month"]
        + "-"
        + frame["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    booking_date = arrival - pd.to_timedelta(frame["lead_time"], unit="D")
    resolved = pd.to_datetime(frame["reservation_status_date"])
    cutoff = booking_date.sort_values().iloc[int(len(frame) * quantile)]
    scheduled_departure = arrival + pd.to_timedelta(
        frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"],
        unit="D",
    )

    train_mask = (
        (booking_date < cutoff)
        & (scheduled_departure < cutoff)
        & (resolved < cutoff)
    )
    test_mask = booking_date >= cutoff
    purged_mask = ~train_mask & ~test_mask

    if not train_mask.any() or not test_mask.any() or (train_mask & test_mask).any():
        raise ValueError("Invalid chronological split")
    if booking_date[train_mask].max() >= booking_date[test_mask].min():
        raise ValueError("Training and test booking periods overlap")
    if resolved[train_mask].max() >= cutoff:
        raise ValueError("Training contains outcomes unresolved at the cutoff")

    return TemporalSplit(cutoff, train_mask, test_mask, purged_mask, booking_date)


def feature_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the explicit pre-outcome feature contract and binary target."""
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[FEATURES].copy(), frame[TARGET].astype(int).copy()
