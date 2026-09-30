import os
import sys
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure project base directory is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from predict import PredictiveMaintenanceModel
from main import app, model_service


class TestXGBoostExperiment(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_rf_production_model_remains_active(self):
        """Verify that live model_service continues to use Random Forest for production inference."""
        self.assertIsNotNone(model_service.model)
        model_class_name = model_service.model.__class__.__name__
        self.assertEqual(model_class_name, "RandomForestRegressor")

    def test_api_model_returns_rf_and_cached_xgboost_results(self):
        """Verify /api/model returns 200 OK with RF metrics and cached XGBoost experiment JSON."""
        res = self.client.get("/api/model")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Check existing Random Forest fields
        self.assertIn("Random Forest", data["production_model"])
        self.assertIn("r2_score", data)
        self.assertIn("mae", data)
        self.assertIn("rmse", data)
        self.assertIn("feature_importance", data)
        self.assertIn("scatter_plot", data)
        self.assertIn("residuals", data)

        # 10-feature RF metric validation
        self.assertGreater(data["r2_score"], 0.85)
        self.assertEqual(len(data["feature_importance"]), 10)

        # Check attached XGBoost experiment payload
        self.assertIn("xgboost_experiment", data)
        xgb_exp = data["xgboost_experiment"]
        if xgb_exp is not None:
            self.assertIn("random_forest", xgb_exp)
            self.assertIn("xgboost", xgb_exp)
            self.assertIn("deltas", xgb_exp)
            self.assertIn("lifecycle_evaluation", xgb_exp)
            self.assertEqual(xgb_exp["xgboost"]["algorithm"], "XGBRegressor")

    def test_api_model_does_not_train_xgboost(self):
        """Verify GET /api/model does NOT invoke XGBoost training on request."""
        with patch("xgboost.XGBRegressor.fit") as mock_fit:
            res = self.client.get("/api/model")
            self.assertEqual(res.status_code, 200)
            mock_fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
