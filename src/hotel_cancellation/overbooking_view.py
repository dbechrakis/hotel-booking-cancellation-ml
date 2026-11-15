"""Read-only view of the committed room-limit results, for the application."""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "outputs" / "overbooking"


def load_limits(directory: Path = OUT) -> tuple[pd.DataFrame, dict]:
    """The recommended extra rooms per night and the counts file with the inferred capacities."""
    limits = pd.read_csv(directory / "limits.csv", parse_dates=["night"])
    counts = json.loads((directory / "counts.json").read_text())
    return limits, counts


def limits_view(
    limits: pd.DataFrame, capacity: int, hotel: str, horizon: int, method: str, cost_ratio: int
) -> pd.DataFrame:
    """One hotel, horizon, method and cost ratio. ``headroom`` is capacity minus the bookings on hand at the horizon."""
    chosen = limits[
        (limits["hotel"] == hotel)
        & (limits["horizon"] == horizon)
        & (limits["method"] == method)
        & (limits["cost_ratio"] == cost_ratio)
    ].sort_values("night")
    out = chosen[["night", "extra_rooms", "on_hand", "constrained"]].copy()
    out["headroom"] = capacity - out["on_hand"]
    return out.reset_index(drop=True)
