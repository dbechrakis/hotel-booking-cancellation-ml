import unittest

import numpy as np
import pandas as pd

from hotel_cancellation.overbooking import (
    crps_discrete,
    eligible_pool,
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


class CountDistributionTests(unittest.TestCase):
    def test_equal_probabilities_give_the_binomial_and_extremes_give_a_point_mass(self):
        pmf = poisson_binomial([0.3] * 5)
        expected = [0.7**5, 5 * 0.3 * 0.7**4, 10 * 0.09 * 0.343, 10 * 0.027 * 0.49, 5 * 0.0081 * 0.7, 0.3**5]
        np.testing.assert_allclose(pmf, expected)
        np.testing.assert_allclose(poisson_binomial([1.0, 1.0, 0.0]), [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            poisson_binomial([1.2])

    def test_crps_is_zero_for_a_correct_point_mass_and_grows_with_distance(self):
        point = np.zeros(11)
        point[4] = 1.0
        self.assertEqual(crps_discrete(point, 4), 0.0)
        self.assertLess(crps_discrete(point, 5), crps_discrete(point, 8))
        self.assertEqual(crps_discrete(point, 7), 3.0)
        self.assertEqual(crps_discrete(point, 15), 11.0)

    def test_interval_covers_the_stated_mass(self):
        pmf = poisson_binomial([0.5] * 40)
        low, high = interval(pmf, 0.8)
        self.assertGreaterEqual(pmf[low : high + 1].sum(), 0.8)
        self.assertLess(pmf[low + 1 : high].sum(), 0.8 + 0.05)
        with self.assertRaises(ValueError):
            interval(pmf, 1.0)

    def test_widen_keeps_the_mean_and_multiplies_the_variance(self):
        pmf = poisson_binomial(np.full(200, 0.5))
        mean, variance = mean_and_variance(pmf)
        wide = widen(pmf, 3.0)
        wide_mean, wide_variance = mean_and_variance(wide)
        self.assertAlmostEqual(wide_mean, mean, places=6)
        self.assertAlmostEqual(wide_variance / variance, 3.0, delta=0.05)
        self.assertEqual(widen(pmf, 1.0).tolist(), pmf.tolist())
        with self.assertRaises(ValueError):
            widen(pmf, 0.5)

    def test_group_cancellations_make_the_exact_distribution_too_narrow_and_widening_repairs_it(self):
        rng = np.random.default_rng(0)
        groups, size, p = 60, 5, 0.7
        nights = 3000
        shows = rng.binomial(groups, p, size=nights) * size
        exact = poisson_binomial(np.full(groups * size, p))
        low, high = interval(exact, 0.8)
        exact_cover = float(np.mean((shows >= low) & (shows <= high)))
        errors = (shows - groups * size * p) ** 2
        _, variance = mean_and_variance(exact)
        factor = overdispersion_factor(errors, np.full(nights, variance))
        wide = widen(exact, factor)
        low, high = interval(wide, 0.8)
        wide_cover = float(np.mean((shows >= low) & (shows <= high)))
        self.assertLess(exact_cover, 0.6)
        self.assertAlmostEqual(wide_cover, 0.8, delta=0.05)


class RoomLimitTests(unittest.TestCase):
    def setUp(self):
        self.on_hand = poisson_binomial([0.5, 0.5])

    def test_a_hand_worked_case(self):
        self.assertEqual(extra_rooms(self.on_hand, 0.5, 2, 1), 3)
        self.assertEqual(extra_rooms(self.on_hand, 0.5, 2, 2), 2)
        self.assertEqual(extra_rooms(self.on_hand, 0.5, 2, 3), 1)

    def test_no_extra_room_when_the_hotel_is_certain_to_be_full(self):
        full = poisson_binomial([1.0] * 5)
        self.assertEqual(extra_rooms(full, 0.9, 4, 2), 0)

    def test_extra_rooms_never_decrease_as_the_cost_ratio_falls(self):
        rooms = [extra_rooms(self.on_hand, 0.6, 2, ratio) for ratio in (4, 3, 2, 1, 0.5)]
        self.assertEqual(rooms, sorted(rooms))
        with self.assertRaises(ValueError):
            extra_rooms(self.on_hand, 0.5, 2, 0)

    def test_scenario_value(self):
        value = scenario_value(4, 0.5, 0.5, 3, 2)
        self.assertEqual(value["extra_shows"], 1.0)
        self.assertEqual(value["gain"], 1.0)
        self.assertEqual(value["walks"], 0.0)
        over = scenario_value(10, 1.0, 1.0, 2, 3)
        self.assertEqual((over["gain"], over["walks"], over["net"]), (2.0, 8.0, 2 - 24.0))
        self.assertEqual(scenario_value(5, 1.0, 1.0, -3, 2)["gain"], 0.0)


class FittingTests(unittest.TestCase):
    def test_calibrator_is_increasing_in_the_score(self):
        rng = np.random.default_rng(1)
        scores = rng.random(2000)
        failed = (rng.random(2000) < scores).astype(int)
        calibrator = horizon_calibrator(scores, failed)
        grid = calibrator.predict(np.linspace(0, 1, 50))
        self.assertTrue(np.all(np.diff(grid) >= 0))

    def test_segment_rates_shrink_small_cells_toward_the_hotel_rate(self):
        rows = [("A", "None", "X", 1)] * 2 + [("A", "None", "Y", 0)] * 400 + [("A", "None", "Y", 1)] * 100
        frame = pd.DataFrame(rows, columns=["hotel", "deposit_type", "market_segment", "failed"])
        rates = segment_rates(frame)
        small = rates["cells"][("A", "None", "X")]
        self.assertGreater(small, rates["hotel"]["A"])
        self.assertLess(small, 1.0)
        self.assertAlmostEqual(small, (2 / 52) * 1 + (50 / 52) * rates["hotel"]["A"])
        probe = pd.DataFrame({"hotel": ["A", "A"], "deposit_type": ["None", "Refundable"], "market_segment": ["X", "Z"]})
        out = segment_failure_probability(rates, probe)
        self.assertAlmostEqual(out[1], rates["hotel"]["A"])

    def test_a_refit_never_uses_a_booking_that_arrives_on_or_after_the_decision_date(self):
        rng = np.random.default_rng(2)
        arrival = pd.Series(pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, 365, 3000), unit="D"))
        for decision in pd.date_range("2017-02-03", "2017-12-20", freq="11D"):
            fitted_on = arrival[eligible_pool(arrival, refit_date(decision))]
            self.assertTrue((fitted_on < decision).all())
            self.assertLessEqual(refit_date(decision), decision)

    def test_overdispersion_factor_needs_enough_nights_and_is_at_least_one(self):
        self.assertEqual(overdispersion_factor([4, 4], [1, 1]), 1.0)
        self.assertEqual(overdispersion_factor([1.0] * 10, [4.0] * 10), 1.0)
        self.assertEqual(overdispersion_factor([8.0] * 10, [4.0] * 10), 2.0)


if __name__ == "__main__":
    unittest.main()
