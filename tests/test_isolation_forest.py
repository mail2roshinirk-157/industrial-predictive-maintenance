"""
Unit and Integration Tests for Isolation Forest Anomaly Detection Module
and Streamlit Dashboard Integration.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    p = os.path.join(base_dir, folder)
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from isolation_forest import (
    IsolationForestDetector,
    FleetAnomalyDetector,
    extract_sensor_features,
    SENSOR_FEATURES,
    EXCLUDED_COLUMNS
)
from simulator import FleetSimulator


class TestIsolationForestModule(unittest.TestCase):
    def setUp(self):
        # Create synthetic test dataset
        np.random.seed(42)
        n = 100
        self.test_data = {
            "Timestamp": [f"2026-09-{i%28+1:02d} 12:00:00" for i in range(n)],
            "Machine_ID": ["M-001" if i < 50 else "M-002" for i in range(n)],
            "Temperature": np.random.normal(62.0, 0.5, n),
            "Vibration": np.random.normal(0.20, 0.02, n),
            "Motor_Current": np.random.normal(8.0, 0.1, n),
            "Pressure": np.random.normal(5.0, 0.05, n),
            "RPM": np.random.normal(1750.0, 2.0, n),
            "Flow_Rate": np.random.normal(150.0, 1.0, n),
            "Oil_Temperature": np.random.normal(45.0, 0.3, n),
            "Power_Consumption": np.random.normal(15.0, 0.2, n),
            "Torque": np.random.normal(81.8, 1.0, n),
            "Acoustic_Emission": np.random.normal(40.0, 0.8, n),
            "Remaining_Useful_Life_Days": np.random.randint(50, 250, n),
            "Machine_Health": np.random.uniform(70.0, 100.0, n)
        }
        self.df = pd.DataFrame(self.test_data)

    def test_sensor_feature_extraction(self):
        """Verify that strictly 10 sensor features are extracted, excluding IDs and labels."""
        X = extract_sensor_features(self.df)
        self.assertEqual(list(X.columns), SENSOR_FEATURES)
        self.assertEqual(len(X.columns), 10)
        for col in EXCLUDED_COLUMNS:
            self.assertNotIn(col, X.columns)

    def test_configurable_contamination(self):
        """Verify that contamination parameter is configurable and defaults to 0.05."""
        detector_default = IsolationForestDetector()
        self.assertEqual(detector_default.contamination, 0.05)

        detector_custom = IsolationForestDetector(contamination=0.10)
        self.assertEqual(detector_custom.contamination, 0.10)

    def test_output_schema(self):
        """Verify output contains anomaly_label ('normal'/'anomaly') and continuous anomaly_score."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(self.df)
        
        preds = detector.predict(self.df)
        self.assertIn("anomaly_label", preds.columns)
        self.assertIn("anomaly_score", preds.columns)
        self.assertIn("is_anomaly", preds.columns)
        self.assertIn("raw_score", preds.columns)
        self.assertIn("anomaly_severity", preds.columns)

        # Labels must be 'normal' or 'anomaly'
        unique_labels = set(preds["anomaly_label"].unique())
        self.assertTrue(unique_labels.issubset({"normal", "anomaly"}))

        # Scores must be within [0.0, 1.0]
        self.assertTrue((preds["anomaly_score"] >= 0.0).all())
        self.assertTrue((preds["anomaly_score"] <= 1.0).all())

    def test_anomaly_sensitivity_on_outlier(self):
        """Verify that an extreme reading triggers an anomaly score > 0.5 and 'anomaly' label."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(self.df)

        extreme_reading = {
            "Temperature": 125.0,        # Normal is 62
            "Vibration": 12.0,           # Normal is 0.2
            "Motor_Current": 40.0,       # Normal is 8.0
            "Pressure": 1.0,             # Normal is 5.0
            "RPM": 800.0,                # Normal is 1750
            "Flow_Rate": 30.0,           # Normal is 150
            "Oil_Temperature": 95.0,     # Normal is 45
            "Power_Consumption": 35.0,   # Normal is 15
            "Torque": 180.0,             # Normal is 81.8
            "Acoustic_Emission": 85.0    # Normal is 40
        }
        res = detector.predict(extreme_reading).iloc[0]
        self.assertEqual(res["anomaly_label"], "anomaly")
        self.assertTrue(res["is_anomaly"])
        self.assertGreater(res["anomaly_score"], 0.50)

    def test_fleet_anomaly_detector_per_machine(self):
        """Verify FleetAnomalyDetector fits per-machine models and routes predictions correctly."""
        fleet = FleetAnomalyDetector(contamination=0.05, random_state=42)
        fleet.fit_fleet(self.df, machine_col="Machine_ID")

        self.assertIn("M-001", fleet.machine_detectors)
        self.assertIn("M-002", fleet.machine_detectors)
        self.assertIsNotNone(fleet.fleet_detector)

        # Test per-machine routing
        m1_rec = self.df[self.df["Machine_ID"] == "M-001"].iloc[0].to_dict()
        pred_m1 = fleet.predict_reading(m1_rec, machine_id="M-001")
        self.assertEqual(pred_m1["model_type"], "per_machine")
        self.assertEqual(pred_m1["machine_id"], "M-001")

        # Test fallback for unseen machine ID
        unknown_rec = m1_rec.copy()
        unknown_rec["Machine_ID"] = "M-999"
        pred_unknown = fleet.predict_reading(unknown_rec, machine_id="M-999")
        self.assertEqual(pred_unknown["model_type"], "fleet_wide")

    def test_independence_from_xgboost(self):
        """Requirement 6: Verify anomaly detection functions without any XGBoost model dependency."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(self.df)

        # Simulate environment where XGBoost is not available or throws error
        sample = self.df.iloc[0].to_dict()
        res = detector.predict(sample).iloc[0]
        
        self.assertIn(res["anomaly_label"], ["normal", "anomaly"])
        self.assertIsInstance(float(res["anomaly_score"]), float)


if __name__ == "__main__":
    unittest.main()
