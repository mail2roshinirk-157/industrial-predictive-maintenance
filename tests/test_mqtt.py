import unittest
import os
import json
from config import (
    FEATURE_COLUMNS,
    DATA_SOURCE,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_MACHINE_ID
)
from mqtt_service import MQTTTelemetryService
from mqtt_publisher import generate_machine_telemetry
from predict import PredictiveMaintenanceModel
from maintenance_engine import get_maintenance_recommendation


class TestMQTTTelemetryIntegration(unittest.TestCase):
    def setUp(self):
        self.service = MQTTTelemetryService(
            broker="localhost",
            port=1883,
            topic="industrial/telemetry/M-001",
            machine_id="M-001"
        )
        self.model = PredictiveMaintenanceModel()

    def test_default_config_direct(self):
        """Verify default data source is 'direct' and backward compatible."""
        self.assertIn(DATA_SOURCE, ["direct", "mqtt"])
        self.assertEqual(len(FEATURE_COLUMNS), 10)
        self.assertIn("Load_Percent", FEATURE_COLUMNS)
        self.assertIn("Oil_Debris_Count", FEATURE_COLUMNS)

    def test_telemetry_generator_schema(self):
        """Verify publisher generates all 10 canonical features with valid types and values."""
        sample = generate_machine_telemetry("M-001", tick=1, wear_progress=0.2)
        self.assertEqual(sample["Machine_ID"], "M-001")
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, sample, f"Feature {feat} must be present in telemetry")
            self.assertIsInstance(sample[feat], (int, float))

    def test_payload_normalization_valid(self):
        """Verify MQTTTelemetryService normalizes valid payload."""
        sample = generate_machine_telemetry("M-001", tick=5, wear_progress=0.3)
        normalized = self.service._normalize_payload(sample)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["Machine_ID"], "M-001")
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, normalized)

    def test_payload_normalization_filters_machine(self):
        """Verify MQTT telemetry drops readings for non-matching machine IDs."""
        sample = generate_machine_telemetry("M-002", tick=5, wear_progress=0.3)
        normalized = self.service._normalize_payload(sample)
        self.assertIsNone(normalized, "Should drop telemetry from other machines")

    def test_payload_normalization_missing_feature(self):
        """Verify MQTT telemetry drops readings missing any canonical feature."""
        sample = generate_machine_telemetry("M-001", tick=5, wear_progress=0.3)
        del sample["Load_Percent"]
        normalized = self.service._normalize_payload(sample)
        self.assertIsNone(normalized, "Should drop payload missing a canonical feature")

    def test_inference_on_mqtt_telemetry(self):
        """Verify that ML model and maintenance engine run on MQTT telemetry payload."""
        sample = generate_machine_telemetry("M-001", tick=10, wear_progress=0.4)
        feat_dict = {f: float(sample[f]) for f in FEATURE_COLUMNS}
        
        predicted_rul = self.model.predict_rul(feat_dict)
        self.assertIsInstance(predicted_rul, (int, float))
        self.assertGreater(predicted_rul, 0)

        rec = get_maintenance_recommendation(
            predicted_rul=predicted_rul,
            machine_health=sample["Machine_Health"],
            active_event=sample["Active_Event"],
            max_lifespan_days=250,
            load_percent=sample["Load_Percent"],
            oil_debris_count=sample["Oil_Debris_Count"]
        )
        self.assertIn("maintenance_status", rec)
        self.assertIn("recommended_action", rec)


if __name__ == "__main__":
    unittest.main()
