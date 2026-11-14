import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("action_replay", Path(__file__).resolve().parents[1] / "scripts" / "action_replay.py")
action_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(action_replay)


def holdout(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.random(n)
    canceled = (rng.random(n) < score).astype(int)
    booking = pd.Timestamp("2017-01-12") + pd.to_timedelta(np.arange(n) // 20, unit="D")
    return pd.DataFrame({"booking_date": booking, "departure": booking + pd.to_timedelta(rng.integers(1, 60, n), unit="D"), "risk_score": score, "is_canceled": canceled})


class ResetTests(unittest.TestCase):
    def test_the_threshold_reaches_the_target_recall_in_its_own_pool(self):
        h = holdout()
        week_end = pd.Timestamp("2017-04-30")
        t = action_replay.reset_candidate(h, week_end)
        pool = h[h["departure"] < week_end].sort_values("booking_date").tail(action_replay.POOL_SIZE)
        recall = ((pool["risk_score"] >= t) & (pool["is_canceled"] == 1)).sum() / pool["is_canceled"].sum()
        self.assertGreaterEqual(recall, 0.80)
        self.assertLess(((pool["risk_score"] > t) & (pool["is_canceled"] == 1)).sum() / pool["is_canceled"].sum(), 0.80 + 0.01)

    def test_outcomes_and_scores_of_stays_that_have_not_ended_cannot_change_the_threshold(self):
        h = holdout()
        week_end = pd.Timestamp("2017-04-30")
        before = action_replay.reset_candidate(h, week_end)
        altered = h.copy()
        open_stays = altered["departure"] >= week_end
        altered.loc[open_stays, "is_canceled"] = 1 - altered.loc[open_stays, "is_canceled"]
        altered.loc[open_stays, "risk_score"] = 1.0
        self.assertEqual(action_replay.reset_candidate(altered, week_end), before)

    def test_a_pool_with_too_few_cancellations_gives_no_reset(self):
        h = holdout()
        h["is_canceled"] = 0
        self.assertIsNone(action_replay.reset_candidate(h, pd.Timestamp("2017-04-30")))


if __name__ == "__main__":
    unittest.main()
