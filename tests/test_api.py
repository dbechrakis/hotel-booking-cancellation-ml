import json
from pathlib import Path
import os
import unittest

import pandas as pd

from hotel_cancellation.model import ModelVersionError, load_model_bundle

try:
    from fastapi.testclient import TestClient

    from hotel_cancellation.api import Booking, MAX_BATCH_SIZE, app
except ImportError as error:  # API extras are optional for the notebook environment.
    if os.environ.get("CI"):
        raise
    raise unittest.SkipTest(f"API dependencies not installed: {error}") from error


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = Booking.model_config["json_schema_extra"]["example"]


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.bundle = load_model_bundle(ROOT / "artifacts")
            cls.metadata = cls.bundle.metadata
        except ModelVersionError as error:
            if os.environ.get("CI"):
                raise
            raise unittest.SkipTest(str(error)) from error
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health_reports_artifact_version(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], self.metadata["artifact_sha256"][:12])
        self.assertIn("X-Process-Time-Ms", response.headers)

    def test_model_card_exposes_holdout_evidence(self):
        card = self.client.get("/model").json()
        self.assertEqual(card["holdout_metrics"], self.metadata["holdout_metrics"])
        self.assertEqual(card["test_rows"], 23989)
        self.assertTrue(card["limitations"])
        self.assertEqual(card["threshold_scale"], "risk_score")
        self.assertLess(
            card["calibration"]["evaluation_calibrated"]["brier"],
            card["calibration"]["evaluation_uncalibrated"]["brier"],
        )

    def test_api_score_matches_direct_artifact_score(self):
        response = self.client.post("/predict", json={"booking": EXAMPLE})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        scores, probabilities = self.bundle.score(pd.DataFrame([EXAMPLE]))
        self.assertAlmostEqual(body["risk_score"], float(scores[0]), places=12)
        self.assertAlmostEqual(body["cancellation_probability"], float(probabilities[0]), places=12)
        self.assertEqual(body["flagged"], scores[0] >= self.metadata["threshold"])
        self.assertEqual(len(body["contributions"]), 6)
        self.assertIsNone(body["decision"])
        self.assertEqual(body["unseen_categories"], {})

    def test_economics_produce_a_decision(self):
        payload = {
            "booking": EXAMPLE,
            "economics": {
                "recoverable_margin": 200,
                "intervention_cost": 5,
                "intervention_success_rate": 0.2,
                "policy_threshold": 0.3,
            },
            "explain": False,
        }
        body = self.client.post("/predict", json=payload).json()
        probability = body["cancellation_probability"]
        self.assertAlmostEqual(
            body["decision"]["assumed_expected_value"], probability * 0.2 * 200 - 5
        )
        self.assertIsNone(body["contributions"])

    def test_batch_scores_align_with_inputs(self):
        long_lead = dict(EXAMPLE, lead_time=400, deposit_type="Non Refund")
        response = self.client.post(
            "/predict/batch", json={"bookings": [EXAMPLE, long_lead]}
        )
        self.assertEqual(response.status_code, 200)
        scores = response.json()["scores"]
        self.assertEqual(len(scores), 2)
        _, expected = self.bundle.score(pd.DataFrame([EXAMPLE, long_lead]))
        for score, probability in zip(scores, expected):
            self.assertAlmostEqual(score["cancellation_probability"], float(probability), places=12)

    def test_unseen_category_is_reported_not_rejected(self):
        booking = dict(EXAMPLE, country="ATLANTIS")
        body = self.client.post("/predict", json={"booking": booking}).json()
        self.assertEqual(body["unseen_categories"], {"country": "ATLANTIS"})

    def test_invalid_bookings_are_rejected(self):
        invalid = [
            dict(EXAMPLE, adults=0),
            dict(EXAMPLE, stays_in_weekend_nights=0, stays_in_week_nights=0),
            dict(EXAMPLE, adr=-1),
            dict(EXAMPLE, arrival_date_month="Smarch"),
            dict(EXAMPLE, reservation_status="Canceled"),
        ]
        for booking in invalid:
            with self.subTest(booking=booking):
                response = self.client.post("/predict", json={"booking": booking})
                self.assertEqual(response.status_code, 422)

    def test_monitoring_tracks_scored_traffic(self):
        before = self.client.get("/monitoring").json()["bookings_scored_total"]
        self.client.post("/predict/batch", json={"bookings": [EXAMPLE] * 250})
        report = self.client.get("/monitoring").json()
        self.assertEqual(report["bookings_scored_total"], before + 250)
        self.assertEqual(report["threshold"], self.metadata["threshold"])
        # 250 identical bookings fill one score decile: a maximal distribution shift.
        self.assertEqual(report["status"], "alert")

    def test_each_score_is_logged_without_booking_attributes(self):
        with self.assertLogs("hotel_cancellation.scores", level="INFO") as logs:
            self.client.post("/predict", json={"booking": EXAMPLE, "explain": False})
        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["event"], "booking_scored")
        self.assertNotIn("country", record)
        self.assertIn("risk_score", record)

    def test_batch_size_is_bounded(self):
        response = self.client.post(
            "/predict/batch", json={"bookings": [EXAMPLE] * (MAX_BATCH_SIZE + 1)}
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
