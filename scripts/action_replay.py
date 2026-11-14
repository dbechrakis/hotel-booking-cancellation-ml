"""Replay the stated monitoring action, as fixed in VALIDATION.md before the run.

When the rolling PSI is at alert in two consecutive weeks, the threshold is re-set to 80% recall on the latest 2,000 bookings
whose scheduled stay had ended. The model and the calibrator are not touched. Writes outputs/monitoring/action_replay.*.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_cancellation.monitoring import drift_status, population_stability_index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "monitoring"
MIN_WEEK_ROWS = 200
TRAILING_WINDOW = 2000
POOL_SIZE = 2000
MIN_POOL_CANCELLATIONS = 300
COOLDOWN_WEEKS = 4
TARGET_RECALL = 0.80


def recall_threshold(scores, outcomes, target_recall: float = TARGET_RECALL) -> float:
    positive_scores = np.sort(scores[outcomes == 1])[::-1]
    return float(positive_scores[int(np.ceil(target_recall * len(positive_scores))) - 1])


def reset_candidate(holdout: pd.DataFrame, week_end: pd.Timestamp):
    """Threshold from the latest POOL_SIZE bookings whose stay had ended by `week_end`, or None when the pool has too few cancellations."""
    matured = holdout[holdout["departure"] < week_end].sort_values("booking_date").tail(POOL_SIZE)
    if int(matured["is_canceled"].sum()) < MIN_POOL_CANCELLATIONS:
        return None
    return recall_threshold(matured["risk_score"].to_numpy(), matured["is_canceled"].to_numpy())


def replay(holdout: pd.DataFrame, reference: dict, deployed_threshold: float) -> dict:
    evaluation = holdout[holdout["calibration_role"] == "evaluation"].sort_values("booking_date").copy()
    evaluation["week"] = evaluation["booking_date"].dt.to_period("W-SUN").dt.start_time
    threshold, last_reset, previous_alert = deployed_threshold, None, False
    rows, events = [], []
    for index, (week, group) in enumerate(evaluation.groupby("week")):
        if len(group) < MIN_WEEK_ROWS:
            continue
        week_end = group["booking_date"].max()
        seen = evaluation[evaluation["booking_date"] <= week_end]
        psi = population_stability_index(reference["bin_shares"], seen["risk_score"].to_numpy()[-TRAILING_WINDOW:], reference["bin_edges"])
        alert = drift_status(psi) == "alert"
        outcome = group["is_canceled"] == 1
        flagged_deployed = group["risk_score"] >= deployed_threshold
        flagged_rule = group["risk_score"] >= threshold
        rows.append({"week": week.date().isoformat(), "bookings": len(group), "threshold_in_force": threshold, "psi_status_alert": alert,
                     "recall_deployed": float((flagged_deployed & outcome).sum() / max(1, outcome.sum())), "recall_rule": float((flagged_rule & outcome).sum() / max(1, outcome.sum())),
                     "precision_deployed": float((flagged_deployed & outcome).sum() / max(1, flagged_deployed.sum())), "precision_rule": float((flagged_rule & outcome).sum() / max(1, flagged_rule.sum())),
                     "flagged_share_deployed": float(flagged_deployed.mean()), "flagged_share_rule": float(flagged_rule.mean())})
        if alert and previous_alert and (last_reset is None or index - last_reset >= COOLDOWN_WEEKS):
            candidate = reset_candidate(holdout, week_end)
            if candidate is None:
                events.append({"week": week.date().isoformat(), "event": "skipped", "reason": "fewer than 300 matured cancellations"})
            else:
                threshold, last_reset = candidate, index
                events.append({"week": week.date().isoformat(), "event": "reset", "threshold": candidate})
        previous_alert = alert
    weekly = pd.DataFrame(rows)
    return {"weekly": weekly, "events": events,
            "summary": {"weeks": len(weekly), "resets": sum(e["event"] == "reset" for e in events), "skips": sum(e["event"] == "skipped" for e in events),
                        "mean_recall_deployed": float(weekly["recall_deployed"].mean()), "mean_recall_rule": float(weekly["recall_rule"].mean()),
                        "weeks_below_80_deployed": int((weekly["recall_deployed"] < 0.8).sum()), "weeks_below_80_rule": int((weekly["recall_rule"] < 0.8).sum()),
                        "mean_precision_deployed": float(weekly["precision_deployed"].mean()), "mean_precision_rule": float(weekly["precision_rule"].mean()),
                        "mean_flagged_share_deployed": float(weekly["flagged_share_deployed"].mean()), "mean_flagged_share_rule": float(weekly["flagged_share_rule"].mean())}}


def main() -> dict:
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    holdout = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv.gz", parse_dates=["booking_date"])
    raw = pd.read_csv(ROOT / "data" / "raw" / "hotel_bookings.csv").loc[holdout["row_id"]].reset_index(drop=True)
    arrival = pd.to_datetime(raw["arrival_date_year"].astype(str) + "-" + raw["arrival_date_month"] + "-" + raw["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    holdout["departure"] = (arrival + pd.to_timedelta(raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"], unit="D")).to_numpy()
    result = replay(holdout, metadata["monitoring_reference"], metadata["threshold"])
    OUT.mkdir(parents=True, exist_ok=True)
    result["weekly"].to_csv(OUT / "action_replay.csv", index=False, float_format="%.4f")
    (OUT / "action_replay.json").write_text(json.dumps({"summary": result["summary"], "events": result["events"]}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    out = main()
    print(out["weekly"].round(3).to_string(index=False))
    print(json.dumps({"summary": out["summary"], "events": out["events"]}, indent=2))
