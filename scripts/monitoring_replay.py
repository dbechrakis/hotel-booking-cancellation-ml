"""Replay the later bookings week by week through the monitor, with hindsight outcomes.

A live service sees only scores. Replaying the 16,835 evaluation bookings in booking order
shows what `/monitoring` would have reported at the end of each week (PSI over the trailing
window of scores, like the API's rolling window), next to that week's realised recall and
precision, which only became known after the stays. Writes outputs/monitoring/.

PSI is computed on a rolling window rather than on each week alone: non-refundable group
bookings arrive in bursts and fill the top score decile, so a single week without them
leaves that decile empty and inflates a per-week PSI on its own.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotel_cancellation.monitoring import PSI_ALERT, PSI_WARN, drift_status, population_stability_index  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "monitoring"
MIN_WEEK_ROWS = 200
TRAILING_WINDOW = 2000


def main() -> None:
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    reference, threshold = metadata["monitoring_reference"], metadata["threshold"]
    holdout = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv.gz", parse_dates=["booking_date"])
    evaluation = holdout[holdout["calibration_role"] == "evaluation"].sort_values("booking_date").copy()
    evaluation["week"] = evaluation["booking_date"].dt.to_period("W-SUN").dt.start_time

    rows = []
    for week, group in evaluation.groupby("week"):
        if len(group) < MIN_WEEK_ROWS:
            continue
        flagged = group["risk_score"] >= threshold
        cancelled = group["is_canceled"] == 1
        seen = evaluation[evaluation["booking_date"] <= group["booking_date"].max()]
        trailing = seen["risk_score"].to_numpy()[-TRAILING_WINDOW:]
        psi = population_stability_index(reference["bin_shares"], trailing, reference["bin_edges"])
        weekly_psi = population_stability_index(reference["bin_shares"], group["risk_score"].to_numpy(), reference["bin_edges"])
        rows.append(
            {
                "week": week.date().isoformat(),
                "bookings": len(group),
                "psi": psi,
                "status": drift_status(psi),
                "single_week_psi": weekly_psi,
                "top_decile_share": float((group["risk_score"] >= reference["bin_edges"][-1]).mean()),
                "flagged_share": flagged.mean(),
                "mean_probability": group["cancellation_probability"].mean(),
                "observed_cancellation_rate": cancelled.mean(),
                "realised_recall": (flagged & cancelled).sum() / max(1, cancelled.sum()),
                "realised_precision": (flagged & cancelled).sum() / max(1, flagged.sum()),
            }
        )
    weekly = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT / "weekly_replay.csv", index=False, float_format="%.4f")

    correlation = float(np.corrcoef(weekly["psi"], weekly["realised_recall"])[0, 1])
    summary = {
        "weeks": len(weekly),
        "bookings": int(weekly["bookings"].sum()),
        "weeks_ok": int((weekly["status"] == "ok").sum()),
        "weeks_warn": int((weekly["status"] == "warn").sum()),
        "weeks_alert": int((weekly["status"] == "alert").sum()),
        "recall_target": 0.8,
        "weeks_below_target_recall": int((weekly["realised_recall"] < 0.8).sum()),
        "psi_recall_correlation": correlation,
        "calibration_gap_mean_abs": float((weekly["mean_probability"] - weekly["observed_cancellation_rate"]).abs().mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    dates = pd.to_datetime(weekly["week"])
    top.plot(dates, weekly["psi"], marker="o", color="#1f4e79", label=f"PSI, trailing {TRAILING_WINDOW:,} scores")
    top.plot(dates, weekly["single_week_psi"], linestyle=":", color="#7f7f7f", label="PSI, single week (noisy)")
    top.axhline(PSI_WARN, color="#d4a017", linestyle="--", linewidth=0.9, label="Warn (0.10)")
    top.axhline(PSI_ALERT, color="#b22222", linestyle="--", linewidth=0.9, label="Alert (0.25)")
    top.set_ylabel("PSI (visible live)")
    top.legend(fontsize=8)
    top.grid(alpha=0.25)
    bottom.plot(dates, weekly["realised_recall"], marker="o", label="Realised recall")
    bottom.plot(dates, weekly["realised_precision"], marker="s", label="Realised precision")
    bottom.plot(dates, weekly["observed_cancellation_rate"], linestyle=":", color="grey", label="Cancellation rate")
    bottom.axhline(0.8, color="black", linestyle="--", linewidth=0.8, label="Recall target")
    bottom.set_ylabel("Known only after stays")
    bottom.legend(fontsize=8, ncol=2)
    bottom.grid(alpha=0.25)
    fig.suptitle("Weekly replay of later bookings through the score monitor")
    fig.tight_layout()
    fig.savefig(OUT / "weekly_replay.png", dpi=150)
    plt.close(fig)
    print(weekly.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
