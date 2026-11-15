import unittest

import pandas as pd

from hotel_cancellation.overbooking_view import limits_view, load_limits


class OverbookingViewTests(unittest.TestCase):
    def test_the_view_filters_one_series_and_computes_headroom(self):
        limits, counts = load_limits()
        capacity = counts["capacity"]["City Hotel"]
        view = limits_view(limits, capacity, "City Hotel", 7, "model_od", 2)
        self.assertEqual(len(view), counts["evaluation_nights"])
        self.assertTrue(view["night"].is_monotonic_increasing)
        self.assertTrue((view["headroom"] == capacity - view["on_hand"]).all())
        self.assertLessEqual(set(view["constrained"]), {0, 1})

    def test_the_view_of_a_hand_built_table(self):
        limits = pd.DataFrame(
            {
                "night": pd.to_datetime(["2017-05-02", "2017-05-01", "2017-05-01"]),
                "hotel": ["A", "A", "B"],
                "horizon": [7, 7, 7],
                "method": ["flat"] * 3,
                "cost_ratio": [2, 2, 2],
                "extra_rooms": [3, 1, 9],
                "on_hand": [95, 90, 50],
                "constrained": [1, 1, 0],
            }
        )
        view = limits_view(limits, 100, "A", 7, "flat", 2)
        self.assertEqual(view["extra_rooms"].tolist(), [1, 3])
        self.assertEqual(view["headroom"].tolist(), [10, 5])


if __name__ == "__main__":
    unittest.main()
