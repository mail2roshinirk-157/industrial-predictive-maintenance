"""
Trains a Gradient Boosting Regressor to replace the existing Random Forest RUL model.

Reuses the exact data pipeline already used by main.py (load_data / preprocess_data),
so this drops in without touching preprocessing.py.

Run this once to produce models/gb_rul_model.pkl, then predict.py will load it instead
of the Random Forest model (see the default model_path change below).
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = os.path.dirname(os.path.abspath(__file__))
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from preprocessing import load_data, preprocess_data

# --- 1. Load data using the existing pipeline ---
df = load_data()
df.ffill(inplace=True)

X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, is_training=True)

# --- 2. Train Gradient Boosting Regressor ---
# n_estimators/learning_rate/max_depth are reasonable defaults for small tabular
# sensor datasets; tune via GridSearchCV/RandomizedSearchCV if you want to squeeze
# out more accuracy later.
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# --- 3. Evaluate ---
y_pred = gb_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Gradient Boosting Regressor")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R2:   {r2:.4f}")

# --- 4. Save model ---
# NOTE: The existing scaler.pkl is reused as-is — it's just a StandardScaler fit on
# [Temperature, Vibration, Motor_Current], which doesn't depend on which regressor
# consumes its output. No need to refit it.
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

model_out_path = os.path.join(models_dir, "gb_rul_model.pkl")
joblib.dump(gb_model, model_out_path)
print(f"Saved trained model to {model_out_path}")
