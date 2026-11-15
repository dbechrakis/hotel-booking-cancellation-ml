"""Overbooking study, as fixed in VALIDATION.md before the run.

Scores the frozen artifact on every booking it did not train on, runs the five methods walk-forward, scores the count
forecasts, applies the room-limit rule and values it under stated demand shares. Writes outputs/overbooking/.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_cancellation.contracts import FEATURES
from hotel_cancellation.data import chronological_split, source_sha256
from hotel_cancellation.ledger import (
    arrival_dates,
    capacity,
    nightly_ledger,
    occupancy_by_night,
    on_hand,
)
from hotel_cancellation.model import load_model_bundle
from hotel_cancellation.overbooking import (
    COST_RATIOS,
    DEMAND_SHARES,
    crps_discrete,
    extra_rooms,
    horizon_calibrator,
    interval,
    mean_and_variance,
    overdispersion_factor,
    poisson_binomial,
    refit_date,
    scenario_value,
    segment_failure_probability,
    segment_rates,
    widen,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUT = ROOT / "outputs" / "overbooking"
EXPECTED_SOURCE_SHA = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
CUTOFF = "2017-01-12"
HOTELS = ("City Hotel", "Resort Hotel")
HORIZONS = (30, 14, 7, 3)
METHODS = ("flat", "segment", "at_booking", "model", "model_od")
SHIFTS = (-5, 0, 5)
FIT_START = pd.Timestamp("2017-02-01")
END = pd.Timestamp("2017-08-31")
PICKUP_NIGHTS = 16
FACTOR_MIN_NIGHTS = 7
CONSTRAINED_SHARE = 0.9
MIN_LATE_BOOKINGS = 10
BOOTSTRAP_RESAMPLES = 1000
SEED = 42
MIDDLE_RATIO = 2


def build_info(frame: pd.DataFrame, bundle, out_of_sample: pd.Series) -> pd.DataFrame:
    arrival = arrival_dates(frame)
    status_date = pd.to_datetime(frame["reservation_status_date"])
    info = pd.DataFrame(
        {
            "hotel": frame["hotel"],
            "arrival": arrival,
            "booking_date": arrival - pd.to_timedelta(frame["lead_time"], unit="D"),
            "cancel_date": status_date.where(frame["reservation_status"] == "Canceled"),
            "nights": frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"],
            "failed": frame["is_canceled"].astype(int),
            "deposit_type": frame["deposit_type"],
            "market_segment": frame["market_segment"],
            "out": out_of_sample,
        }
    )
    info["risk_score"] = np.nan
    info["deployed_cancel_probability"] = np.nan
    scored = frame.loc[out_of_sample, FEATURES]
    scores = bundle.pipeline.predict_proba(scored)[:, 1]
    info.loc[out_of_sample, "risk_score"] = scores
    info.loc[out_of_sample, "deployed_cancel_probability"] = bundle.calibrator.predict(scores)
    return info


def alive_at(info: pd.DataFrame, horizon: int) -> pd.Series:
    gap = pd.Timedelta(days=horizon)
    return (info["booking_date"] <= info["arrival"] - gap) & (
        info["cancel_date"].isna() | (info["cancel_date"] > info["arrival"] - gap)
    )


class Fits:
    def __init__(self, info: pd.DataFrame):
        self.info = info
        self.cache: dict = {}
        self.alive = {h: alive_at(info, h) for h in HORIZONS}

    def get(self, refit: pd.Timestamp, horizon: int) -> dict:
        key = (refit, horizon)
        if key not in self.cache:
            info = self.info
            pool = info[info["out"] & (info["nights"] > 0) & (info["arrival"] < refit) & self.alive[horizon]]
            self.cache[key] = {
                "flat": 1 - pool.groupby("hotel")["failed"].mean(),
                "segments": segment_rates(pool),
                "calibrator": horizon_calibrator(pool["risk_score"].to_numpy(), pool["failed"].to_numpy()),
            }
        return self.cache[key]


def show_probabilities(method: str, rows: pd.DataFrame, fit: dict, hotel: str) -> np.ndarray:
    if method == "flat":
        return np.full(len(rows), float(fit["flat"][hotel]))
    if method == "segment":
        return 1 - segment_failure_probability(fit["segments"], rows)
    if method == "at_booking":
        return 1 - rows["deployed_cancel_probability"].to_numpy()
    return 1 - fit["calibrator"].predict(rows["risk_score"].to_numpy())


def pickup_counts(ledger: pd.DataFrame, horizon: int) -> pd.Series:
    late = ledger[ledger["shows"] & (ledger["booking_date"] > ledger["night"] - pd.Timedelta(days=horizon))]
    return late.groupby(["hotel", "night"]).size()


def later_booking_stats(ledger: pd.DataFrame, horizon: int) -> pd.DataFrame:
    late = ledger[ledger["booking_date"] > ledger["night"] - pd.Timedelta(days=horizon)]
    return late.groupby(["hotel", "night"])["shows"].agg(["sum", "count"])


def pickup_distribution(counts: pd.Series, hotel: str, night: pd.Timestamp, horizon: int) -> np.ndarray:
    decision = night - pd.Timedelta(days=horizon)
    series = counts[hotel]
    past = series[(series.index < decision) & (series.index.dayofweek == night.dayofweek)].tail(PICKUP_NIGHTS)
    values = past.to_numpy(dtype=int)
    if len(values) == 0:
        return np.array([1.0])
    return np.bincount(values) / len(values)


def forecasts_for(ledger_groups, info: pd.DataFrame, fits: Fits, hotel: str, horizon: int, evaluation_start: pd.Timestamp):
    nights = pd.date_range(FIT_START + pd.Timedelta(days=horizon), END)
    per_night = {}
    for night in nights:
        sub = ledger_groups.get((hotel, night))
        if sub is None:
            continue
        held = on_hand(sub, night, hotel, horizon)
        rows = info.loc[held["booking_id"].to_numpy()]
        fit = fits.get(refit_date(night - pd.Timedelta(days=horizon)), horizon)
        pmfs, q = {}, {}
        for method in ("flat", "segment", "at_booking", "model"):
            p = show_probabilities(method, rows, fit, hotel)
            pmfs[method] = poisson_binomial(p)
            q[method] = float(p.mean()) if len(p) else 0.0
        q["model_od"] = q["model"]
        per_night[night] = {"pmfs": pmfs, "q": q, "observed": int(held["shows"].sum()), "on_hand": len(held)}
    model_stats = {n: mean_and_variance(v["pmfs"]["model"]) for n, v in per_night.items()}
    for night, entry in per_night.items():
        decision = night - pd.Timedelta(days=horizon)
        past = [m for m in per_night if m < decision]
        errors = [(per_night[m]["observed"] - model_stats[m][0]) ** 2 for m in past]
        variances = [model_stats[m][1] for m in past]
        factor = overdispersion_factor(errors, variances, FACTOR_MIN_NIGHTS)
        entry["factor"] = factor
        entry["pmfs"]["model_od"] = widen(entry["pmfs"]["model"], factor)
    return {n: v for n, v in per_night.items() if n >= evaluation_start}


def block_bootstrap(weeks: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    labels = np.unique(weeks)
    sums = np.array([values[weeks == w].sum() for w in labels])
    counts = np.array([(weeks == w).sum() for w in labels])
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(labels), size=(BOOTSTRAP_RESAMPLES, len(labels)))
    draws = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high), float(draws.std(ddof=1))


def score_table(nights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(nights["night"])
    weeks = (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).to_numpy()
    nights = nights.assign(week=weeks, error=nights["mean"] - nights["observed"])
    for method in METHODS:
        for horizon in HORIZONS:
            for group in (*HOTELS, "All"):
                chosen = nights[(nights["method"] == method) & (nights["horizon"] == horizon)]
                if group != "All":
                    chosen = chosen[chosen["hotel"] == group]
                row = {"method": method, "horizon": horizon, "hotel": group, "nights": len(chosen)}
                for name, column in (("bias", "error"), ("crps", "crps"), ("coverage80", "in80"), ("coverage95", "in95")):
                    values = chosen[column].to_numpy(dtype=float)
                    low, high, _ = block_bootstrap(chosen["week"].to_numpy(), values)
                    row[name] = float(values.mean())
                    row[f"{name}_low"] = low
                    row[f"{name}_high"] = high
                rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(nights: pd.DataFrame, horizon: int, first: str, second: str) -> dict:
    wide = nights[nights["horizon"] == horizon].pivot_table(index=["night", "hotel"], columns="method", values="crps").reset_index()
    dates = pd.to_datetime(wide["night"])
    weeks = (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).to_numpy()
    values = (wide[second] - wide[first]).to_numpy(dtype=float)
    low, high, se = block_bootstrap(weeks, values)
    return {"second_minus_first": float(values.mean()), "low": low, "high": high, "standard_error": se}


def run() -> dict:
    started = time.time()
    if source_sha256(SOURCE) != EXPECTED_SOURCE_SHA:
        raise ValueError("The source file does not match the recorded SHA-256")
    counts = json.loads((OUT / "counts.json").read_text())
    evaluation_start = pd.Timestamp(counts["evaluation_start"])
    frame = pd.read_csv(SOURCE)
    split = chronological_split(frame)
    bundle = load_model_bundle(ROOT / "artifacts")
    info = build_info(frame, bundle, ~split.train_mask)
    ledger = nightly_ledger(frame)
    capacities = {h: capacity(ledger, h, CUTOFF) for h in HOTELS}
    if capacities != counts["capacity"]:
        raise ValueError("The capacities differ from counts.json")
    groups = {key: sub for key, sub in ledger.groupby(["hotel", "night"])}
    fits = Fits(info)
    occupancy = {h: occupancy_by_night(ledger, h) for h in HOTELS}
    night_rows, limit_rows, scenario_rows = [], [], []
    per_night_all = {}
    for horizon in HORIZONS:
        counts_pickup = pickup_counts(ledger, horizon)
        late = later_booking_stats(ledger, horizon)
        for hotel in HOTELS:
            per_night = forecasts_for(groups, info, fits, hotel, horizon, evaluation_start)
            per_night_all[(hotel, horizon)] = per_night
            hotel_late = late.loc[hotel].loc[evaluation_start:END]
            hotel_rate = float(hotel_late["sum"].sum() / hotel_late["count"].sum())
            for night, entry in per_night.items():
                for method in METHODS:
                    pmf = entry["pmfs"][method]
                    mean, variance = mean_and_variance(pmf)
                    low80, high80 = interval(pmf, 0.8)
                    low95, high95 = interval(pmf, 0.95)
                    observed = entry["observed"]
                    night_rows.append(
                        {
                            "night": night.date().isoformat(),
                            "hotel": hotel,
                            "horizon": horizon,
                            "method": method,
                            "on_hand": entry["on_hand"],
                            "mean": mean,
                            "variance": variance,
                            "observed": observed,
                            "crps": crps_discrete(pmf, observed),
                            "low80": low80,
                            "high80": high80,
                            "low95": low95,
                            "high95": high95,
                            "in80": int(low80 <= observed <= high80),
                            "in95": int(low95 <= observed <= high95),
                            "overdispersion_factor": entry["factor"] if method == "model_od" else 1.0,
                        }
                    )
            for method in METHODS:
                for shift in SHIFTS:
                    cap = capacities[hotel] + shift
                    ratios = COST_RATIOS if shift == 0 else (MIDDLE_RATIO,)
                    for ratio in ratios:
                        extras, constrained = [], []
                        for night, entry in per_night.items():
                            is_constrained = entry["on_hand"] >= CONSTRAINED_SHARE * cap
                            pickup = pickup_distribution(counts_pickup, hotel, night, horizon)
                            total = np.convolve(entry["pmfs"][method], pickup)
                            k = extra_rooms(total, entry["q"][method], cap, ratio)
                            extras.append(k)
                            constrained.append(is_constrained)
                            if shift == 0:
                                limit_rows.append(
                                    {
                                        "night": night.date().isoformat(),
                                        "hotel": hotel,
                                        "horizon": horizon,
                                        "method": method,
                                        "cost_ratio": ratio,
                                        "extra_rooms": k,
                                        "on_hand": entry["on_hand"],
                                        "constrained": int(is_constrained),
                                    }
                                )
                        for share in DEMAND_SHARES:
                            gain = walks = net = 0.0
                            taken_nights = 0
                            for night, k, flag in zip(per_night, extras, constrained, strict=True):
                                if not flag or k == 0:
                                    continue
                                taken_nights += 1
                                stats = late.loc[(hotel, night)] if (hotel, night) in late.index else None
                                show_rate = hotel_rate
                                if stats is not None and stats["count"] >= MIN_LATE_BOOKINGS:
                                    show_rate = float(stats["sum"] / stats["count"])
                                free = cap - int(occupancy[hotel].get(night, 0))
                                value = scenario_value(k, share, show_rate, free, ratio)
                                gain += value["gain"]
                                walks += value["walks"]
                                net += value["net"]
                            n = len(per_night)
                            scenario_rows.append(
                                {
                                    "hotel": hotel,
                                    "horizon": horizon,
                                    "method": method,
                                    "cost_ratio": ratio,
                                    "demand_share": share,
                                    "capacity_shift": shift,
                                    "nights": n,
                                    "constrained_nights": int(sum(constrained)),
                                    "nights_with_extra_sold": taken_nights,
                                    "gain": gain,
                                    "walks": walks,
                                    "net": net,
                                }
                            )
        print(f"horizon {horizon} done in {time.time() - started:.0f}s", flush=True)
    nights_frame = pd.DataFrame(night_rows)
    limits = pd.DataFrame(limit_rows)
    scenarios = pd.DataFrame(scenario_rows)
    scores = score_table(nights_frame)
    OUT.mkdir(parents=True, exist_ok=True)
    nights_frame.to_csv(OUT / "night_forecasts.csv", index=False)
    scores.to_csv(OUT / "forecast_scores.csv", index=False)
    limits.to_csv(OUT / "limits.csv", index=False)
    scenarios.to_csv(OUT / "scenarios.csv", index=False)
    summary = summarise(scores, nights_frame, limits, scenarios, counts)
    summary["runtime_seconds"] = time.time() - started
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def scenario_per_100(scenarios: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon", "method", "cost_ratio", "demand_share", "capacity_shift"]
    grouped = scenarios.groupby(keys)[["nights", "gain", "walks", "net", "constrained_nights", "nights_with_extra_sold"]].sum().reset_index()
    for column in ("gain", "walks", "net"):
        grouped[f"{column}_per_100_nights"] = grouped[column] / grouped["nights"] * 100
    return grouped


def grade(scores: pd.DataFrame, nights: pd.DataFrame, limits: pd.DataFrame) -> dict:
    def cell(method, horizon, hotel="All"):
        return scores[(scores["method"] == method) & (scores["horizon"] == horizon) & (scores["hotel"] == hotel)].iloc[0]

    graded = {}
    biases = {f"{h}_{k}": float(cell("at_booking", k, h)["bias"]) for h in HOTELS for k in HORIZONS}
    city7 = biases["City Hotel_7"]
    graded["O1"] = {"passed": bool(city7 < -5 and all(v < 0 for v in biases.values())), "values": biases}
    o2 = {k: {"model": float(cell("model", k)["crps"]), "flat": float(cell("flat", k)["crps"])} for k in (30, 14)}
    graded["O2"] = {"passed": all(v["model"] < v["flat"] for v in o2.values()), "values": o2}
    diff = paired_difference(nights, 30, "model", "segment")
    graded["O3"] = {
        "passed": bool(diff["second_minus_first"] > diff["standard_error"]),
        "values": {"segment_minus_model_crps": diff["second_minus_first"], "bootstrap_standard_error": diff["standard_error"], "low": diff["low"], "high": diff["high"]},
    }
    o4 = float(cell("model", 30)["coverage80"])
    graded["O4"] = {"passed": o4 < 0.70, "values": {"coverage80_model_30": o4}}
    o5 = {k: float(cell("model_od", k)["coverage80"]) for k in HORIZONS}
    graded["O5"] = {"passed": all(0.70 <= v <= 0.90 for v in o5.values()), "values": o5}
    chosen = limits[(limits["method"] == "model_od") & (limits["horizon"] == 30) & (limits["cost_ratio"] == MIDDLE_RATIO) & (limits["constrained"] == 1)]
    share = float((chosen["extra_rooms"] >= 1).mean()) if len(chosen) else 0.0
    graded["O6"] = {"passed": share > 0.5, "values": {"share_of_constrained_nights_with_an_extra_room": share, "constrained_nights": int(len(chosen))}}
    graded["O7"] = {"passed": None, "values": "see scenarios.csv"}
    return graded


def summarise(scores, nights, limits, scenarios, counts) -> dict:
    per_100 = scenario_per_100(scenarios)
    return {
        "data_source": "Hotel Booking Demand, public file with the recorded SHA-256",
        "capacity": counts["capacity"],
        "evaluation_start": counts["evaluation_start"],
        "evaluation_nights": counts["evaluation_nights"],
        "expectations": grade(scores, nights, limits),
        "forecast_scores_all_hotels": scores[scores["hotel"] == "All"].to_dict(orient="records"),
        "scenario_per_100_nights_both_hotels": per_100.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(json.dumps(run()["expectations"], indent=2))
