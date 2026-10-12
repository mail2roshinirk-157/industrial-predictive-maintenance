"""
train_gb_model.py
=================
Trains a Gradient Boosting Regressor on the v2 10-feature fleet dataset.
Uses the unified config.py single source of truth.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)

from config import MODELS_DIR, FEATURE_COLUMNS
from preprocessing import load_data, preprocess_data

def train_gb():
    print("Loading v2 dataset...")
    df = load_data()
    df.ffill(inplace=True)

    print("Preprocessing with unified 10-feature schema...")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, is_training=True)

    print("Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)

    y_pred = gb_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Gradient Boosting Regressor Performance:")
    print(f"  MAE:  {mae:.4f} Days")
    print(f"  RMSE: {rmse:.4f} Days")
    print(f"  R2:   {r2:.6f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_out_path = os.path.join(MODELS_DIR, "gb_rul_model.pkl")
    joblib.dump(gb_model, model_out_path)
    print(f"Saved trained model to {model_out_path}")
    return mae, rmse, r2

if __name__ == "__main__":
    train_gb()
