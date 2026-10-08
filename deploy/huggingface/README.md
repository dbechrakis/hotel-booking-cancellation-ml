---
title: Hotel Cancellation Risk API
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Calibrated cancellation-risk scoring API with drift monitor
---

# Hotel Cancellation Risk API

Interactive documentation for the scoring service from
[dbechrakis/hotel-booking-cancellation-ml](https://github.com/dbechrakis/hotel-booking-cancellation-ml).
This Space is rebuilt from that repository whenever the service changes.

- `/docs`: try `POST /predict` with the example booking
- `/model`: validation evidence and limitations
- `/monitoring`: drift of recent scores against the threshold window

A portfolio demonstration on a public historical dataset. It has no authentication.
