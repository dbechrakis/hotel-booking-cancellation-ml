# Overbooking method

This page explains the room limit in `src/hotel_cancellation/overbooking.py` and where it stops being valid. The study that uses it is in `scripts/overbooking_backtest.py`, and its design and result are in `VALIDATION.md`.

## The rule

The hotel earns r for a room it fills and pays w for a guest it cannot house. Selling one more room above capacity C adds r when that guest fits and costs w when the hotel is already full. Let S be the number of guests who arrive. Selling the k-th extra room is worth it while the chance that it leads to a walk stays at or below r over r plus w. This is the critical ratio of the newsvendor problem.

1. Selling k extra rooms means S_k is the shows from the bookings on hand, plus the pickup still to come, plus k more bookings that each show with some probability.
2. The k-th extra room leads to a walk when S_k reaches C plus 1 or more.
3. The expected change in value from selling room k is r times the chance it fits minus w times the chance it walks, to a good approximation.
4. That is positive while the chance of reaching C plus 1 is below r over r plus w.
5. The chance of reaching C plus 1 rises with k, so the rule is a threshold. Accept rooms one by one and stop at the first one that fails.
6. With r equal to 1 and a cost ratio w over r of 2, the critical ratio is one third.

`extra_rooms` implements this. The search stops at 40 extra rooms.

## The count distribution

Among the bookings on hand, each shows with its own probability. The number that show is a Poisson binomial variable, computed exactly by convolution. The exact distribution assumes independence. A group that cancels together breaks it, and real counts are wider. `widen` multiplies the variance by a factor by convolving with a discretised normal. Mass pushed below 0 or above the number on hand is folded into the end points.

The factor used in the study is estimated from past nights as the ratio of squared error to predicted variance. Because squared error includes the bias, the factor also absorbs the bias. The result of the study shows that this over-widens the distribution.

## Pickup

The bookings still to come are taken from the empirical distribution of pickup at the same horizon, for the same hotel and weekday, over the 16 most recent same-weekday nights before the decision date.

## Scenario value

The extra rooms that would have been booked are a stated share of the rule's recommendation on constrained nights. They show with the realised show rate of that night's later bookings. The extra guests fill the rooms that were in fact free. The rest are walked. The value is gains minus the cost ratio times walks, in units of one room night.

## Limits

- Capacity is the highest occupancy seen before 2017-01-12, not a published room count.
- The data records no turned-away demand. The demand share is an assumption and the result depends on it.
- Room types are ignored. A hotel overbooks by room type.
- Walks are priced at a fixed multiple of a room night. Effects on reviews and repeat business are not modelled.
- The study covers 139 nights for each of two hotels in one spring and summer. Adjacent nights share bookings.
- The frozen model was built to rank bookings at booking time. A model built for survival from a given day might do better. The study does not test that.
