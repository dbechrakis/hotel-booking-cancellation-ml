import unittest

import numpy as np
import pandas as pd

from hotel_cancellation.data import split_at

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    lead = rng.integers(0, 120, n)
    arrival = days + pd.to_timedelta(lead, unit="D")
    stay = rng.integers(1, 6, n)
    resolved = arrival + pd.to_timedelta(stay, unit="D")
    return pd.DataFrame({"arrival_date_year": arrival.year, "arrival_date_month": [MONTHS[m - 1] for m in arrival.month], "arrival_date_day_of_month": arrival.day,
                         "lead_time": lead, "stays_in_weekend_nights": 0, "stays_in_week_nights": stay, "reservation_status_date": resolved.strftime("%Y-%m-%d")})


class SplitAtTests(unittest.TestCase):
    def test_no_training_row_is_known_after_the_cutoff_and_test_rows_are_the_next_six_months(self):
        frame = synthetic()
        cutoff = pd.Timestamp("2016-03-01")
        split = split_at(frame, cutoff)
        arrival = pd.to_datetime(frame["arrival_date_year"].astype(str) + "-" + frame["arrival_date_month"] + "-" + frame["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
        departure = arrival + pd.to_timedelta(frame["stays_in_week_nights"], unit="D")
        self.assertTrue((split.booking_date[split.train_mask] < cutoff).all())
        self.assertTrue((departure[split.train_mask] < cutoff).all())
        self.assertTrue((pd.to_datetime(frame["reservation_status_date"])[split.train_mask] < cutoff).all())
        window = split.booking_date[split.test_mask]
        self.assertTrue((window >= cutoff).all() and (window < cutoff + pd.DateOffset(months=6)).all())
        self.assertFalse((split.train_mask & split.test_mask).any())
        self.assertEqual(int(split.purged_mask.sum()), len(frame) - int(split.train_mask.sum()) - int(split.test_mask.sum()))

    def test_changing_bookings_after_the_cutoff_cannot_change_the_training_rows(self):
        frame = synthetic()
        cutoff = pd.Timestamp("2016-03-01")
        before = split_at(frame, cutoff).train_mask.copy()
        later = split_at(frame, cutoff).test_mask
        altered = frame.copy()
        altered.loc[later, "lead_time"] = altered.loc[later, "lead_time"] + 1
        self.assertTrue(split_at(altered, cutoff).train_mask.equals(before))


if __name__ == "__main__":
    unittest.main()
