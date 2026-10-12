import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np

# Ensure project base directory is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from simulator import FleetSimulator, RealTimeMachineSimulator
from predict import PredictiveMaintenanceModel
from maintenance_engine import get_maintenance_recommendation
from main import app, sim_state, generate_next_telemetry_step


class TestFleetSimulator(unittest.TestCase):
    def setUp(self):
        self.fleet = FleetSimulator(max_lifespan_days=250, degradation_factor=1.8)

    def test_eight_machines_created(self):
        """1. Verify fleet contains exactly 8 machines."""
        self.assertEqual(len(self.fleet.machines), 8)
        expected_ids = [f"M-00{i}" for i in range(1, 9)]
        self.assertCountEqual(list(self.fleet.machines.keys()), expected_ids)

    def test_synchronized_step(self):
        """Verify all 8 machines advance together on a single tick."""
        records = self.fleet.step()
        self.assertEqual(len(records), 8)
        self.assertEqual(self.fleet.tick, 1)

        for rec in records:
            self.assertEqual(rec["Tick"], 1)
            self.assertIn("Machine_ID", rec)
            self.assertIn("Temperature", rec)
            self.assertIn("Vibration", rec)
            self.assertIn("Motor_Current", rec)
            self.assertIn("Machine_Health", rec)

    def test_heterogeneous_telemetry(self):
        """Verify machines produce distinct non-identical telemetry."""
        records = self.fleet.step()
        temps = [r["Temperature"] for r in records]
        self.assertGreater(len(set(temps)), 1)


class TestBatchInference(unittest.TestCase):
    def setUp(self):
        self.model_service = PredictiveMaintenanceModel()

    def test_batch_prediction_single_model_call(self):
        """7 & 8. Verify exactly ONE model.predict() call receives an 8-row x 10-feature matrix."""
        sample_batch = [
            {
                "Temperature": 62.0 + i,
                "Vibration": 0.20 + (i * 0.05),
                "Motor_Current": 8.0 + (i * 0.1),
                "Pressure": 5.0 + (i * 0.1),
                "RPM": 1750.0 - (i * 10),
                "Flow_Rate": 150.0 - i,
                "Oil_Temperature": 45.0 + i,
                "Power_Consumption": 15.0 + (i * 0.5),
                "Load_Percent": 70.0 + i,
                "Oil_Debris_Count": 5 + i
            }
            for i in range(8)
        ]

        if self.model_service.model is not None:
            with patch.object(self.model_service.model, 'predict', wraps=self.model_service.model.predict) as mock_predict:
                preds = self.model_service.predict_rul_batch(sample_batch)

                # Exactly ONE call to model.predict
                self.assertEqual(mock_predict.call_count, 1)

                args, kwargs = mock_predict.call_args
                X_input = args[0]
                self.assertEqual(X_input.shape[0], 8)
                self.assertEqual(X_input.shape[1], 10)

                self.assertEqual(len(preds), 8)
                for p in preds:
                    self.assertIsInstance(p, int)
                    self.assertGreaterEqual(p, 0)
        else:
            self.skipTest("Model file not found; skipping mock inference test.")

    def test_machine_ids_mapping(self):
        """9. Verify predictions map 1:1 to machine IDs in sequential order."""
        records = generate_next_telemetry_step()
        self.assertEqual(len(records), 8)
        m_ids = [r["Machine_ID"] for r in records]
        expected_ids = [f"M-00{i}" for i in range(1, 9)]
        self.assertEqual(m_ids, expected_ids)
        for r in records:
            self.assertIn("Predicted_RUL", r)
            self.assertIsInstance(r["Predicted_RUL"], int)


class TestMaintenanceEngineIndicators(unittest.TestCase):
    def test_explainable_indicator_generation(self):
        """6. Verify explainable abnormal indicators are generated from real telemetry baseline deviations."""
        # High temperature deviation scenario (78.4°C vs baseline 62.0°C)
        res = get_maintenance_recommendation(
            predicted_rul=60,
            machine_health=55.0,
            temperature=78.4,
            vibration=0.20,
            motor_current=8.0,
            pressure=5.0,
            rpm=1750.0,
            flow_rate=150.0,
            oil_temperature=45.0,
            power_consumption=15.0,
            temp_base=62.0,
            vib_base=0.20,
            curr_base=8.0,
            press_base=5.0,
            rpm_base=1750.0,
            flow_base=150.0,
            oil_temp_base=45.0,
            power_base=15.0
        )
        indicators = res.get("abnormal_indicators", [])
        self.assertGreater(len(indicators), 0)
        self.assertTrue(any("Elevated Temperature" in ind for ind in indicators))

        # Nominal scenario (all at baselines)
        res_nom = get_maintenance_recommendation(
            predicted_rul=200,
            machine_health=98.0,
            temperature=62.0,
            vibration=0.20,
            motor_current=8.0,
            pressure=5.0,
            rpm=1750.0,
            flow_rate=150.0,
            oil_temperature=45.0,
            power_consumption=15.0,
            temp_base=62.0,
            vib_base=0.20,
            curr_base=8.0,
            press_base=5.0,
            rpm_base=1750.0,
            flow_base=150.0,
            oil_temp_base=45.0,
            power_base=15.0
        )
        self.assertEqual(len(res_nom.get("abnormal_indicators", [])), 0)


class TestFleetKPIAndAPIContracts(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_dynamic_fleet_kpis(self):
        """2, 3, 4, 5. Verify total machines, status/priority counts, avg health, and avg RUL are derived dynamically."""
        res = self.client.get("/api/current")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("fleet_kpi", data)
        kpi = data["fleet_kpi"]
        machines = data["machines"]

        # 2. Total machines derived dynamically
        self.assertEqual(kpi["total_machines"], len(machines))
        self.assertEqual(kpi["total_machines"], 8)

        # 3. Status/Priority counts match machine statuses
        healthy_expected = sum(1 for m in machines if m["Machine_Status"] == "Healthy")
        self.assertEqual(kpi["healthy_count"], healthy_expected)

        # 4. Avg health matches mean of machine healths
        avg_health_expected = round(float(np.mean([m["Machine_Health"] for m in machines])), 1)
        self.assertEqual(kpi["avg_health"], avg_health_expected)

        # 5. Avg RUL matches mean of predicted RULs
        avg_rul_expected = int(round(float(np.mean([m["Predicted_RUL"] for m in machines]))))
        self.assertEqual(kpi["avg_predicted_rul"], avg_rul_expected)

    def test_production_vs_experiment_model_roles(self):
        """10 & 11. Verify RF remains production model and XGBoost is offline experiment only."""
        status_res = self.client.get("/api/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertIn("Random Forest", status_res.json()["model"])

        model_res = self.client.get("/api/model")
        self.assertEqual(model_res.status_code, 200)
        model_data = model_res.json()
        self.assertIn("Random Forest", model_data.get("production_model"))
        self.assertIn("xgboost_experiment", model_data)

    def test_existing_api_endpoints_functional(self):
        """12 & 13. Verify all existing API endpoints remain fully functional."""
        endpoints = [
            "/api/status",
            "/api/current",
            "/api/machines",
            "/api/machines/M-001",
            "/api/history?machine_id=M-001",
            "/api/maintenance",
            "/api/model"
        ]
        for ep in endpoints:
            res = self.client.get(ep)
            self.assertEqual(res.status_code, 200, f"Endpoint {ep} failed with status {res.status_code}")


if __name__ == "__main__":
    unittest.main()
