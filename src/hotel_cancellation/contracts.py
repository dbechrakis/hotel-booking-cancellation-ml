"""Stable feature and data contracts for training and inference."""

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
]
TARGET = "is_canceled"

# The deployed contract leaves out `country`. In this dataset the guest's country is often only confirmed at check-in, so a
# cancelled booking can keep the hotel's default country, and the input may carry the outcome (README, "Why country was removed").
# The notebook's comparison and the tracked experiments were run with it and keep using these lists as recorded history.
REMOVED_AFTER_REVIEW = ["country"]
NOTEBOOK_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES[:3] + ["country"] + CATEGORICAL_FEATURES[3:]
NOTEBOOK_FEATURES = FEATURES[:10] + ["country"] + FEATURES[10:]

# arrival_date_year is known before the outcome, but as a numeric input it extrapolates:
# every later booking falls beyond the training years and scores were inflated by about
# 14 points. Seasonality stays in the month, week and day features.
EXCLUDED_EXTRAPOLATING_FIELDS = {"arrival_date_year"}

EXCLUDED_LEAKAGE_FIELDS = {
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "total_of_special_requests",
    "required_car_parking_spaces",
}
