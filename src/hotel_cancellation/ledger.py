"""Room ledger: which bookings hold a room on which night, and what was known on a given day."""

import numpy as np
import pandas as pd

LEDGER_COLUMNS = ["booking_id", "hotel", "night", "booking_date", "cancel_date", "shows"]


def arrival_dates(frame: pd.DataFrame) -> pd.Series:
    """Arrival date of each booking."""
    return pd.to_datetime(
        frame["arrival_date_year"].astype(str)
        + "-"
        + frame["arrival_date_month"]
        + "-"
        + frame["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )


def nightly_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per booking and night it covers. Zero-night bookings are left out.

    ``cancel_date`` is the cancellation date of a cancelled booking. A no-show has none, so it stays on hand until the night.
    ``shows`` is true when the booking was not cancelled and did not fail to arrive."""
    nights = (frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]).astype(int)
    arrival = arrival_dates(frame)
    booking_date = arrival - pd.to_timedelta(frame["lead_time"], unit="D")
    status_date = pd.to_datetime(frame["reservation_status_date"])
    cancelled = frame["reservation_status"] == "Canceled"
    keep = nights > 0
    repeats = nights[keep].to_numpy()
    index = frame.index[keep]
    offsets = np.arange(int(repeats.sum())) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    base = arrival[keep].repeat(repeats).reset_index(drop=True)
    ledger = pd.DataFrame(
        {
            "booking_id": index.repeat(repeats),
            "hotel": frame.loc[keep, "hotel"].repeat(repeats).to_numpy(),
            "night": (base + pd.to_timedelta(offsets, unit="D")).to_numpy(),
            "booking_date": booking_date[keep].repeat(repeats).to_numpy(),
            "cancel_date": status_date.where(cancelled)[keep].repeat(repeats).to_numpy(),
            "shows": (frame.loc[keep, "is_canceled"] == 0).repeat(repeats).to_numpy(),
        }
    )
    return ledger[LEDGER_COLUMNS].reset_index(drop=True)


def _night_rows(ledger: pd.DataFrame, night, hotel: str) -> pd.DataFrame:
    return ledger[(ledger["night"] == pd.Timestamp(night)) & (ledger["hotel"] == hotel)]


def on_hand(ledger: pd.DataFrame, night, hotel: str, horizon: int) -> pd.DataFrame:
    """Bookings that cover ``night``, were made by the decision date and were not cancelled by it."""
    decision = pd.Timestamp(night) - pd.Timedelta(days=horizon)
    rows = _night_rows(ledger, night, hotel)
    alive = rows["cancel_date"].isna() | (rows["cancel_date"] > decision)
    return rows[(rows["booking_date"] <= decision) & alive]


def pickup(ledger: pd.DataFrame, night, hotel: str, horizon: int) -> pd.DataFrame:
    """Bookings made after the decision date that cover ``night`` and show up."""
    decision = pd.Timestamp(night) - pd.Timedelta(days=horizon)
    rows = _night_rows(ledger, night, hotel)
    return rows[(rows["booking_date"] > decision) & rows["shows"]]


def occupied(ledger: pd.DataFrame, night, hotel: str) -> int:
    """Rooms actually occupied on ``night``."""
    return int(_night_rows(ledger, night, hotel)["shows"].sum())


def occupancy_by_night(ledger: pd.DataFrame, hotel: str) -> pd.Series:
    """Rooms occupied on every night of one hotel."""
    rows = ledger[(ledger["hotel"] == hotel) & ledger["shows"]]
    return rows.groupby("night").size()


def capacity(ledger: pd.DataFrame, hotel: str, before) -> int:
    """Largest number of rooms occupied on any night before ``before``. An inference, not a published room count."""
    counts = occupancy_by_night(ledger, hotel)
    return int(counts[counts.index < pd.Timestamp(before)].max())
