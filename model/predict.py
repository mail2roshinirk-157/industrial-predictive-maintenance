import joblib
import pandas as pd
import numpy as np
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from config import FEATURE_COLUMNS, MODELS_DIR

class PredictiveMaintenanceModel:
    """
    Predictive maintenance inference engine using v2 feature schema (10 features).
    Loads models/rf_rul_model.pkl + models/scaler.pkl by default.
    """
    def __init__(self, model_path=None, scaler_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        def resolve_path(p, default_name):
            if p is None:
                p = os.path.join(MODELS_DIR, default_name)
            if os.path.exists(p):
                return p
            p_from_cur = os.path.join(current_dir, p)
            if os.path.exists(p_from_cur):
                return p_from_cur
            p_from_parent = os.path.join(os.path.dirname(current_dir), p)
            if os.path.exists(p_from_parent):
                return p_from_parent
            return p

        resolved_model = resolve_path(model_path, "rf_rul_model.pkl")
        resolved_scaler = resolve_path(scaler_path, "scaler.pkl")

        if os.path.exists(resolved_model) and os.path.exists(resolved_scaler):
            self.model = joblib.load(resolved_model)
            self.scaler = joblib.load(resolved_scaler)
        else:
            self.model = None
            self.scaler = None

    def predict_rul(self, temperature, vibration, current, pressure=5.0, rpm=1750.0, flow_rate=150.0, oil_temperature=45.0, power_consumption=15.0, load_percent=70.0, oil_debris_count=5):
        preds = self.predict_rul_batch([{
            "Temperature": temperature,
            "Vibration": vibration,
            "Motor_Current": current,
            "Pressure": pressure,
            "RPM": rpm,
            "Flow_Rate": flow_rate,
            "Oil_Temperature": oil_temperature,
            "Power_Consumption": power_consumption,
            "Load_Percent": load_percent,
            "Oil_Debris_Count": oil_debris_count
        }])
        return preds[0] if preds else 0

    def predict_rul_batch(self, records_or_df):
        """
        Executes TRUE BATCH ML INFERENCE for all machines in a single pass.
        Accepts a DataFrame or list of dicts containing all 10 telemetry features.
        Executes EXACTLY ONE scaler.transform() and ONE model.predict() call for the entire fleet batch.
        Returns a list of integer RUL predictions.
        """
        if not self.model or not self.scaler:
            if isinstance(records_or_df, list):
                return [0] * len(records_or_df)
            return [0] * len(records_or_df.index)

        if isinstance(records_or_df, list):
            df = pd.DataFrame(records_or_df)
        else:
            df = records_or_df.copy()

        # If Load_Percent or Oil_Debris_Count are missing in records, apply reasonable physical defaults
        if "Load_Percent" not in df.columns:
            df["Load_Percent"] = 70.0
            
        if "Oil_Debris_Count" not in df.columns:
            if "Machine_Health" in df.columns:
                p_val = np.clip(1.0 - df["Machine_Health"] / 100.0, 0.0, 1.0)
                df["Oil_Debris_Count"] = np.random.poisson(5.0 + 400.0 * (p_val ** 3))
            else:
                df["Oil_Debris_Count"] = 5

        feature_cols = FEATURE_COLUMNS
        # Fill missing features if any record lacks any feature
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X_features = df[feature_cols].ffill().bfill().fillna(0.0)
        X_scaled = self.scaler.transform(X_features)
        raw_preds = self.model.predict(X_scaled)
        return [max(0, int(round(p))) for p in raw_preds]


def filter_displayed_rul(raw_prediction, prev_displayed_rul=None, health=100.0):
    """
    Temporal smoothing filter for displayed Remaining Useful Life (RUL).
    - Monotonic non-increasing.
    - Smoothly decrements (1–3 days per step based on ML prediction and degradation rate).
    - Preserves gradual day-by-day degradation while tracking raw model predictions closely.
    - Converges smoothly to 0 when machine reaches terminal state (health = 0%).
    Returns (val_float, displayed_rul_int).
    """
    raw_pred = float(raw_prediction)
    
    if prev_displayed_rul is None:
        val = max(0.0, raw_pred)
        return val, int(round(val))
        
    prev = float(prev_displayed_rul)
    
    # Strictly enforce terminal failure state
    if float(health) <= 0.0:
        return 0.0, 0
        
    # Terminal countdown acceleration near zero
    if prev <= 3.0:
        val = max(0.0, prev - 1.0)
        return val, int(round(val))
        
    # Monotonic degradation logic:
    if raw_pred >= prev:
        # Physical wear monotonic constraint
        decay = 1.0 if float(health) > 60.0 else 2.0
        val = max(0.0, prev - decay)
    else:
        # Track accelerated degradation smoothly
        step_gap = prev - raw_pred
        if step_gap > 10.0:
            step_down = 3.0
        elif step_gap > 3.0:
            step_down = 2.0
        else:
            step_down = 1.0
        val = max(0.0, prev - step_down)
        
    return val, int(round(val))


def get_future_trend(historical_records, n_future_steps=30, steps=None, **kwargs):
    """
    Extrapolates physical degradation trends into future steps.
    Supports both:
    1. 1D sequence of float values (e.g., history_window for a single sensor).
    2. List of dict records / DataFrame with multiple columns.
    """
    if steps is not None:
        n_future_steps = steps
    elif "n_future_steps" in kwargs:
        n_future_steps = kwargs["n_future_steps"]
        
    if historical_records is None:
        return []

    # Case 1: 1D list/array of numerical values (used by main.py make_trend_dict)
    if isinstance(historical_records, (list, tuple, np.ndarray)):
        if len(historical_records) == 0:
            return []
        if isinstance(historical_records[0], (int, float, np.number)):
            if len(historical_records) < 2:
                return [round(float(historical_records[-1]), 2)] * n_future_steps if historical_records else [0.0] * n_future_steps

            x = np.arange(len(historical_records))
            y = np.array(historical_records, dtype=float)
            last_val = float(y[-1])

            window = min(30, len(historical_records))
            deg = 2 if window >= 10 else 1
            try:
                coef = np.polyfit(x[-window:], y[-window:], deg)
                poly1d_fn = np.poly1d(coef)
            except Exception:
                coef = np.polyfit(x, y, 1)
                poly1d_fn = np.poly1d(coef)

            future_x = np.arange(len(historical_records) - 1, len(historical_records) - 1 + n_future_steps)
            base_future = poly1d_fn(future_x)

            offset = last_val - base_future[0]
            base_future = base_future + offset

            recent_residuals = y[-window:] - poly1d_fn(x[-window:])
            std_dev = float(np.std(recent_residuals)) if len(recent_residuals) > 1 else 0.5
            std_dev = max(0.15, min(std_dev, 2.5))

            np.random.seed(42)
            future_y = [round(last_val, 2)]
            for i in range(1, n_future_steps):
                uncertainty_factor = (i / n_future_steps) * 0.75
                noise = float(np.random.normal(0, std_dev * uncertainty_factor))
                future_y.append(round(float(base_future[i] + noise), 2))

            return future_y

    # Case 2: List of record dicts or DataFrame
    df_hist = pd.DataFrame(historical_records)
    n_hist = len(df_hist)

    if n_hist < 2:
        return {}

    future_trends = {}

    for col in FEATURE_COLUMNS:
        if col in df_hist.columns:
            vals = df_hist[col].tolist()
            future_trends[col] = get_future_trend(vals, n_future_steps=n_future_steps)

    return future_trends
