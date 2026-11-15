import unittest

import pandas as pd

from hotel_cancellation.ledger import (
    capacity,
    nightly_ledger,
    occupied,
    on_hand,
    pickup,
)

NIGHT = "2017-05-10"


def booking(hotel, arrival, nights, lead, status, status_date):
    year, month, day = arrival.split("-")
    month_name = pd.Timestamp(arrival).strftime("%B")
    return {
        "hotel": hotel,
        "arrival_date_year": int(year),
        "arrival_date_month": month_name,
        "arrival_date_day_of_month": int(day),
        "lead_time": lead,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": nights,
        "reservation_status": status,
        "reservation_status_date": status_date,
        "is_canceled": int(status != "Check-Out"),
    }


def make_frame():
    rows = {
        "A": booking("Resort Hotel", "2017-05-09", 2, 30, "Check-Out", "2017-05-11"),
        "B": booking("Resort Hotel", "2017-05-10", 1, 5, "Check-Out", "2017-05-11"),
        "C": booking("Resort Hotel", "2017-05-10", 3, 20, "Canceled", "2017-05-01"),
        "D": booking("Resort Hotel", "2017-05-10", 1, 40, "No-Show", "2017-05-10"),
        "E": booking("Resort Hotel", "2017-05-08", 5, 10, "Check-Out", "2017-05-13"),
        "F": booking("Resort Hotel", "2017-05-10", 0, 10, "Check-Out", "2017-05-10"),
        "G": booking("City Hotel", "2017-05-10", 1, 50, "Check-Out", "2017-05-11"),
        "H": booking("Resort Hotel", "2017-05-10", 1, 2, "Check-Out", "2017-05-11"),
        "I": booking("Resort Hotel", "2017-05-10", 1, 10, "Canceled", "2017-05-10"),
    }
    frame = pd.DataFrame(rows.values(), index=range(len(rows)))
    frame.attrs["names"] = list(rows)
    return frame


def names(frame, ledger_rows):
    labels = frame.attrs["names"]
    return sorted(labels[i] for i in ledger_rows["booking_id"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.ledger = nightly_ledger(self.frame)

    def test_ledger_has_one_row_per_booking_and_night_and_drops_zero_night_bookings(self):
        self.assertEqual(len(self.ledger), 2 + 1 + 3 + 1 + 5 + 1 + 1 + 1)
        self.assertNotIn(5, set(self.ledger["booking_id"]))
        rows = self.ledger[self.ledger["booking_id"] == 4]
        self.assertEqual(list(rows["night"]), [pd.Timestamp("2017-05-08") + pd.Timedelta(days=i) for i in range(5)])

    def test_cancelled_bookings_carry_their_date_and_no_shows_carry_none(self):
        cancelled = self.ledger[self.ledger["booking_id"] == 2]
        self.assertTrue((cancelled["cancel_date"] == pd.Timestamp("2017-05-01")).all())
        self.assertFalse(cancelled["shows"].any())
        no_show = self.ledger[self.ledger["booking_id"] == 3]
        self.assertTrue(no_show["cancel_date"].isna().all())
        self.assertFalse(no_show["shows"].any())

    def test_occupied_counts_only_guests_who_showed(self):
        self.assertEqual(occupied(self.ledger, NIGHT, "Resort Hotel"), 4)
        self.assertEqual(occupied(self.ledger, NIGHT, "City Hotel"), 1)

    def test_on_hand_at_each_horizon(self):
        expected = {
            14: ["A", "C", "D"],
            7: ["A", "D", "E", "I"],
            3: ["A", "B", "D", "E", "I"],
        }
        for horizon, labels in expected.items():
            rows = on_hand(self.ledger, NIGHT, "Resort Hotel", horizon)
            self.assertEqual(names(self.frame, rows), labels, horizon)

    def test_pickup_is_the_later_bookings_that_show(self):
        self.assertEqual(names(self.frame, pickup(self.ledger, NIGHT, "Resort Hotel", 7)), ["B", "H"])
        self.assertEqual(names(self.frame, pickup(self.ledger, NIGHT, "Resort Hotel", 3)), ["H"])

    def test_on_hand_shows_plus_pickup_is_the_occupancy_when_nothing_is_lost_in_between(self):
        held = on_hand(self.ledger, NIGHT, "Resort Hotel", 7)
        late = pickup(self.ledger, NIGHT, "Resort Hotel", 7)
        self.assertEqual(int(held["shows"].sum()) + len(late), occupied(self.ledger, NIGHT, "Resort Hotel"))

    def test_capacity_is_the_largest_occupancy_before_the_date(self):
        self.assertEqual(capacity(self.ledger, "Resort Hotel", "2017-05-10"), 2)
        self.assertEqual(capacity(self.ledger, "Resort Hotel", "2017-05-11"), 4)

    def test_changing_outcomes_after_the_decision_date_cannot_change_what_was_on_hand(self):
        horizon = 7
        before = on_hand(self.ledger, NIGHT, "Resort Hotel", horizon)
        changed = self.frame.copy()
        changed.loc[4, ["reservation_status", "reservation_status_date", "is_canceled"]] = ["Canceled", "2017-05-05", 1]
        changed.loc[1, ["reservation_status", "reservation_status_date", "is_canceled"]] = ["Canceled", "2017-05-06", 1]
        after = on_hand(nightly_ledger(changed), NIGHT, "Resort Hotel", horizon)
        self.assertEqual(sorted(before["booking_id"]), sorted(after["booking_id"]))
        self.assertEqual(sorted(before["booking_date"]), sorted(after["booking_date"]))


if __name__ == "__main__":
    unittest.main()
