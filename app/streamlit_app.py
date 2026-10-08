"""Interactive cancellation-risk and threshold decision application."""

from pathlib import Path

import pandas as pd
import streamlit as st

from hotel_cancellation.contracts import FEATURES
from hotel_cancellation.decision import (
    expected_intervention_value,
    recommendation,
    risk_band,
    threshold_table,
)
from hotel_cancellation.model import load_model_bundle, local_contributions


ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_resources():
    bundle = load_model_bundle(ROOT / "artifacts")
    holdout = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv.gz")
    return bundle, holdout


def categorical_input(metadata: dict, feature: str, label: str) -> str:
    options = metadata["category_options"][feature]
    default = metadata["categorical_defaults"][feature]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)


def numeric_default(metadata: dict, feature: str) -> float:
    return float(metadata["numeric_defaults"][feature])


def booking_form(metadata: dict) -> dict:
    """Collect the explicit 22-feature booking contract."""
    with st.expander("Booking and stay", expanded=True):
        left, middle, right = st.columns(3)
        with left:
            hotel = categorical_input(metadata, "hotel", "Hotel type")
            lead_time = st.number_input(
                "Lead time (days)", min_value=0, max_value=800,
                value=int(numeric_default(metadata, "lead_time")),
            )
            adr = st.number_input(
                "Average daily rate", min_value=0.0, max_value=1000.0,
                value=max(0.0, numeric_default(metadata, "adr")), step=5.0,
            )
            reserved_room_type = categorical_input(
                metadata, "reserved_room_type", "Reserved room type"
            )
        with middle:
            arrival_date_year = st.selectbox("Arrival year", [2015, 2016, 2017], index=1)
            arrival_date_month = categorical_input(
                metadata, "arrival_date_month", "Arrival month"
            )
            arrival_date_week_number = st.number_input(
                "Arrival week", min_value=1, max_value=53,
                value=max(1, int(numeric_default(metadata, "arrival_date_week_number"))),
            )
            arrival_date_day_of_month = st.number_input(
                "Arrival day", min_value=1, max_value=31,
                value=max(1, int(numeric_default(metadata, "arrival_date_day_of_month"))),
            )
        with right:
            stays_in_weekend_nights = st.number_input(
                "Weekend nights", min_value=0, max_value=30,
                value=int(numeric_default(metadata, "stays_in_weekend_nights")),
            )
            stays_in_week_nights = st.number_input(
                "Week nights", min_value=0, max_value=60,
                value=max(1, int(numeric_default(metadata, "stays_in_week_nights"))),
            )
            adults = st.number_input(
                "Adults", min_value=0, max_value=20,
                value=max(1, int(numeric_default(metadata, "adults"))),
            )
            children = st.number_input(
                "Children", min_value=0, max_value=10,
                value=max(0, int(numeric_default(metadata, "children"))),
            )
            babies = st.number_input(
                "Babies", min_value=0, max_value=10,
                value=max(0, int(numeric_default(metadata, "babies"))),
            )

    with st.expander("Customer and channel", expanded=False):
        left, middle, right = st.columns(3)
        with left:
            meal = categorical_input(metadata, "meal", "Meal")
            country = categorical_input(metadata, "country", "Country")
            market_segment = categorical_input(metadata, "market_segment", "Market segment")
        with middle:
            distribution_channel = categorical_input(
                metadata, "distribution_channel", "Distribution channel"
            )
            deposit_type = categorical_input(metadata, "deposit_type", "Deposit type")
            customer_type = categorical_input(metadata, "customer_type", "Customer type")
        with right:
            is_repeated_guest = st.selectbox("Repeated guest", [0, 1], index=0)
            previous_cancellations = st.number_input(
                "Previous cancellations", min_value=0, max_value=30,
                value=int(numeric_default(metadata, "previous_cancellations")),
            )
            previous_bookings_not_canceled = st.number_input(
                "Previous completed bookings", min_value=0, max_value=100,
                value=int(numeric_default(metadata, "previous_bookings_not_canceled")),
            )

    return {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "adr": adr,
    }


def main() -> None:
    st.set_page_config(page_title="Hotel Cancellation Decision System", layout="wide")
    st.title("Hotel Cancellation Risk Decision System")
    st.caption(
        "Chronologically validated risk ranking → transparent intervention economics"
    )
    st.warning(
        "Portfolio decision support—not a production policy or causal uplift model. "
        "Economic outputs depend entirely on the assumptions you enter."
    )

    bundle, holdout = load_resources()
    metadata = bundle.metadata
    scoring_tab, policy_tab, evidence_tab = st.tabs(
        ["Score a booking", "Threshold simulator", "Model evidence"]
    )

    with scoring_tab:
        inputs = booking_form(metadata)
        st.subheader("Intervention assumptions")
        c1, c2, c3, c4 = st.columns(4)
        recoverable_margin = c1.number_input(
            "Recoverable margin", min_value=0.0, value=100.0, step=10.0
        )
        intervention_cost = c2.number_input(
            "Intervention cost", min_value=0.0, value=5.0, step=1.0
        )
        success_rate = c3.slider(
            "Intervention success rate", 0.0, 1.0, 0.25, 0.05
        )
        policy_threshold = c4.slider("Policy threshold (risk score)", 0.10, 0.90, 0.50, 0.05)

        if st.button("Score booking", type="primary"):
            booking = pd.DataFrame([inputs], columns=FEATURES)
            scores, probabilities = bundle.score(booking)
            score, probability = float(scores[0]), float(probabilities[0])
            value = expected_intervention_value(
                probability, recoverable_margin, intervention_cost, success_rate
            )
            action = recommendation(score, policy_threshold, value)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk score", f"{score:.2f}")
            m2.metric("Calibrated cancellation probability", f"{probability:.1%}")
            m3.metric("Risk band", risk_band(probability))
            m4.metric("Expected intervention value", f"{value:,.2f}")
            st.info(action)
            st.caption(
                "The risk score ranks bookings and is compared with the policy threshold. "
                "The calibrated probability is what the expected value multiplies, because the "
                "class-weighted score overstates cancellation rates."
            )
            st.markdown("#### Largest model contributions")
            contributions = local_contributions(bundle.pipeline, booking)
            st.dataframe(contributions, use_container_width=True, hide_index=True)
            st.caption(
                "Contributions explain this logistic score. They are associations, "
                "not proof that changing a field will prevent cancellation."
            )

    with policy_tab:
        st.subheader("Historical holdout policy simulator")
        p1, p2, p3 = st.columns(3)
        policy_margin = p1.number_input(
            "Recoverable margin per prevented cancellation",
            min_value=0.0, value=100.0, step=10.0, key="policy_margin",
        )
        policy_cost = p2.number_input(
            "Cost per intervention", min_value=0.0, value=5.0, step=1.0,
            key="policy_cost",
        )
        policy_success = p3.slider(
            "Assumed intervention success", 0.0, 1.0, 0.25, 0.05,
            key="policy_success",
        )
        selected_threshold = st.slider(
            "Decision threshold (risk score)", 0.10, 0.90, 0.50, 0.05,
            key="simulator_threshold",
        )
        table = threshold_table(
            holdout["is_canceled"],
            holdout["risk_score"],
            policy_margin,
            policy_cost,
            policy_success,
        )
        selected = table.iloc[(table["threshold"] - selected_threshold).abs().argmin()]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Bookings contacted", f"{int(selected['interventions']):,}")
        q2.metric("Precision", f"{selected['precision']:.1%}")
        q3.metric("Cancellation recall", f"{selected['recall']:.1%}")
        q4.metric("Assumed net value", f"{selected['assumed_net_value']:,.0f}")
        st.line_chart(table.set_index("threshold")["assumed_net_value"])
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        st.caption(
            "Net value = identified cancellations × assumed success × recoverable margin "
            "− all interventions × cost. This is a scenario, not observed uplift."
        )

    with evidence_tab:
        st.subheader("Validated evidence")
        metrics = metadata["holdout_metrics"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Holdout rows", f"{metadata['test_rows']:,}")
        e2.metric("F1 at 0.50", f"{metrics['f1']:.3f}")
        e3.metric("Average precision", f"{metrics['average_precision']:.3f}")
        e4.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        calibration = metadata["calibration"]
        before, after = calibration["evaluation_uncalibrated"], calibration["evaluation_calibrated"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Brier score", f"{after['brier']:.3f}", f"{after['brier'] - before['brier']:+.3f}", delta_color="inverse")
        c2.metric("Calibration error (ECE)", f"{after['ece']:.3f}", f"{after['ece'] - before['ece']:+.3f}", delta_color="inverse")
        c3.metric("Mean predicted vs observed", f"{after['mean_predicted']:.1%} vs {after['observed_rate']:.1%}")
        st.caption(
            f"Isotonic calibration was fitted on bookings before {calibration['calibration_cutoff']} "
            f"({calibration['calibration_rows']:,}) and evaluated on the {calibration['evaluation_rows']:,} later ones."
        )
        st.image(str(ROOT / "outputs" / "calibration_reliability.png"), width=480)
        st.markdown(
            f"Training uses bookings before **{metadata['booking_date_cutoff']}** whose "
            "scheduled stays and outcomes had matured before that cutoff. Later bookings "
            "form the holdout."
        )
        st.json(
            {
                "model": metadata["model"],
                "train_rows": metadata["train_rows"],
                "test_rows": metadata["test_rows"],
                "purged_rows": metadata["purged_rows"],
                "limitations": metadata["limitations"],
            }
        )


main()
