"""Counts for the overbooking study. Feasibility only. No forecast is scored here.

Writes outputs/overbooking/counts.json: capacities, evaluation nights, the gate counts and the constrained nights.
"""

import json
from pathlib import Path

import pandas as pd

from hotel_cancellation.data import chronological_split, source_sha256
from hotel_cancellation.ledger import (
    arrival_dates,
    capacity,
    nightly_ledger,
    occupancy_by_night,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
CUTOFF = "2017-01-12"
HORIZONS = (30, 14, 7, 3)
HOTELS = ("City Hotel", "Resort Hotel")
EVALUATION_START = "2017-04-01"
EVALUATION_END = "2017-08-31"
GATE_BOOKINGS = 1000
GATE_FAILURES = 50
MAX_SHIFTS = 2
CONSTRAINED_SHARE = 0.9


def gate_counts(frame: pd.DataFrame, out_of_sample: pd.Series, start: pd.Timestamp) -> list[dict]:
    """At the first decision date of each horizon, bookings alive at that horizon that arrived before it, per hotel."""
    arrival = arrival_dates(frame)
    booking_date = arrival - pd.to_timedelta(frame["lead_time"], unit="D")
    status_date = pd.to_datetime(frame["reservation_status_date"])
    cancel_date = status_date.where(frame["reservation_status"] == "Canceled")
    rows = []
    for horizon in HORIZONS:
        decision = start - pd.Timedelta(days=horizon)
        alive_at_own_horizon = (booking_date <= arrival - pd.Timedelta(days=horizon)) & (
            cancel_date.isna() | (cancel_date > arrival - pd.Timedelta(days=horizon))
        )
        pool = out_of_sample & (arrival < decision) & alive_at_own_horizon
        for hotel in HOTELS:
            chosen = pool & (frame["hotel"] == hotel)
            rows.append(
                {
                    "horizon": horizon,
                    "hotel": hotel,
                    "first_decision_date": decision.date().isoformat(),
                    "alive_bookings": int(chosen.sum()),
                    "later_failed_to_show": int(frame.loc[chosen, "is_canceled"].sum()),
                }
            )
    return rows


def passes(rows: list[dict]) -> bool:
    return all(r["alive_bookings"] >= GATE_BOOKINGS and r["later_failed_to_show"] >= GATE_FAILURES for r in rows)


def main() -> dict:
    digest = source_sha256(SOURCE)
    if digest != EXPECTED_SOURCE_SHA:
        raise ValueError("The source file does not match the recorded SHA-256")
    frame = pd.read_csv(SOURCE)
    split = chronological_split(frame)
    ledger = nightly_ledger(frame)
    capacities = {hotel: capacity(ledger, hotel, CUTOFF) for hotel in HOTELS}
    start = pd.Timestamp(EVALUATION_START)
    shifts = 0
    gate = gate_counts(frame, ~split.train_mask, start)
    initial_gate = gate
    while not passes(gate) and shifts < MAX_SHIFTS:
        shifts += 1
        start = start + pd.Timedelta(days=14)
        gate = gate_counts(frame, ~split.train_mask, start)
    gate_passed = passes(gate)
    nights = pd.date_range(start, EVALUATION_END)
    constrained = []
    for hotel in HOTELS:
        by_night = ledger[ledger["hotel"] == hotel]
        for horizon in HORIZONS:
            decision = by_night["booking_date"] <= by_night["night"] - pd.Timedelta(days=horizon)
            alive = by_night["cancel_date"].isna() | (by_night["cancel_date"] > by_night["night"] - pd.Timedelta(days=horizon))
            held = by_night[decision & alive].groupby("night").size().reindex(nights, fill_value=0)
            constrained.append(
                {
                    "hotel": hotel,
                    "horizon": horizon,
                    "constrained_nights": int((held >= CONSTRAINED_SHARE * capacities[hotel]).sum()),
                    "nights": len(nights),
                }
            )
    occupancy = {hotel: occupancy_by_night(ledger, hotel) for hotel in HOTELS}
    record = {
        "source_sha256": digest,
        "bookings": len(frame),
        "checked_out": int((frame["reservation_status"] == "Check-Out").sum()),
        "cancelled": int((frame["reservation_status"] == "Canceled").sum()),
        "no_show": int((frame["reservation_status"] == "No-Show").sum()),
        "zero_night_bookings": int(((frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]) == 0).sum()),
        "ledger_rows": len(ledger),
        "cutoff": CUTOFF,
        "capacity": capacities,
        "p95_occupancy_before_cutoff": {
            hotel: float(occupancy[hotel][occupancy[hotel].index < pd.Timestamp(CUTOFF)].quantile(0.95)) for hotel in HOTELS
        },
        "evaluation_start": start.date().isoformat(),
        "evaluation_end": EVALUATION_END,
        "evaluation_nights": len(nights),
        "start_shifts": shifts,
        "initial_gate": initial_gate,
        "gate_passed": gate_passed,
        "gate": gate,
        "constrained_nights": constrained,
    }
    out = ROOT / "outputs" / "overbooking" / "counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
