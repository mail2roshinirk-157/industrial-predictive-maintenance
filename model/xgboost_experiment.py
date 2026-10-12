import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project modules are discoverable
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)
for folder in ['preprocessing', 'model', 'utils', 'simulator']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)

from config import FEATURE_COLUMNS, BASELINE_3FEATURE_COLUMNS, DATASET_PATH, MODELS_DIR
from preprocessing import load_data, preprocess_data

FEATURE_NAMES = FEATURE_COLUMNS

def run_xgboost_experiment():
    print("==================================================")
    print("      INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM  ")
    print("     10-Feature RF Production vs XGBoost Experiment")
    print("==================================================")

    # 1. Load Dataset & Preprocess on all 10 Features
    print(f"\n[1/5] Loading training dataset ({DATASET_PATH})...")
    df = load_data()
    print(f"Dataset shape: {df.shape} ({len(df)} total records)")

    lifecycle_col = "Machine_Health" if "Machine_Health" in df.columns else None

    print("\n[2/5] Applying preprocessing & standard scaling (10 features, test_size=0.2, random_state=42)...")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, is_training=True)
    print(f"Train size: {len(X_train_scaled)} samples | Test size: {len(X_test_scaled)} samples")

    # 2. Train & Evaluate Preserved 3-Feature RF Baseline
    print("\n[2b/5] Training/Evaluating 3-Feature RF Baseline (Temperature, Vibration, Motor_Current)...")
    base_rf_path = os.path.join(MODELS_DIR, "rf_rul_model_3feature_baseline.pkl")
    base_scaler_path = os.path.join(MODELS_DIR, "scaler_3feature_baseline.pkl")
    
    X3_tr = df.loc[y_train.index, BASELINE_3FEATURE_COLUMNS].values
    X3_te = df.loc[y_test.index, BASELINE_3FEATURE_COLUMNS].values
    s3 = StandardScaler()
    X3_tr_scaled = s3.fit_transform(X3_tr)
    X3_te_scaled = s3.transform(X3_te)
    joblib.dump(s3, base_scaler_path)

    m3 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m3.fit(X3_tr_scaled, y_train.values)
    joblib.dump(m3, base_rf_path)

    y3_pred = m3.predict(X3_te_scaled)
    baseline_3feat_metrics = {
        "r2_score": round(float(r2_score(y_test, y3_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y3_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y3_pred))), 2)
    }
    print(f"  3-Feature RF Baseline -> R²: {baseline_3feat_metrics['r2_score']} | MAE: {baseline_3feat_metrics['mae']}d | RMSE: {baseline_3feat_metrics['rmse']}d")

    # 3. Train & Evaluate 10-Feature Production Random Forest
    print("\n[3/5] Evaluating Production 10-Feature Random Forest (rf_rul_model.pkl)...")
    rf_model_path = os.path.join(MODELS_DIR, "rf_rul_model.pkl")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, rf_model_path)

    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))

    rf_importances = [round(float(imp), 4) for imp in rf_model.feature_importances_]
    print(f"  10-Feature RF Production -> R²: {rf_r2:.4f} | MAE: {rf_mae:.2f} Days | RMSE: {rf_rmse:.2f} Days")

    # 4. Train & Evaluate 10-Feature XGBoost Regressor
    print("\n[4/5] Training 10-Feature XGBoost Regressor (n_estimators=500, lr=0.05, max_depth=6)...")
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)

    xgb_model_path = os.path.join(MODELS_DIR, "xgb_rul_model.pkl")
    joblib.dump(xgb_model, xgb_model_path)
    print(f"  Saved XGBoost model binary to: {xgb_model_path}")

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    xgb_r2 = float(r2_score(y_test, y_pred_xgb))
    xgb_mae = float(mean_absolute_error(y_test, y_pred_xgb))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

    xgb_importances = [round(float(imp), 4) for imp in xgb_model.feature_importances_]
    print(f"  10-Feature XGBoost Exp   -> R²: {xgb_r2:.4f} | MAE: {xgb_mae:.2f} Days | RMSE: {xgb_rmse:.2f} Days")

    # Metric Deltas (XGBoost vs RF Production)
    delta_r2 = float(round(xgb_r2 - rf_r2, 4))
    delta_mae = float(round(xgb_mae - rf_mae, 2))
    delta_rmse = float(round(xgb_rmse - rf_rmse, 2))

    # 5. Degradation Lifecycle Evaluation
    print("\n[5/5] Conducting Degradation-Lifecycle Validation across Machine Health regions...")
    lifecycle_results = []
    
    if lifecycle_col and lifecycle_col in df.columns:
        test_health = df.loc[y_test.index, lifecycle_col].values
        
        regions = [
            ("Early Degradation (Health >= 80%)", test_health >= 80.0),
            ("Mid Degradation (60% <= Health < 80%)", (test_health >= 60.0) & (test_health < 80.0)),
            ("Late Degradation (40% <= Health < 60%)", (test_health >= 40.0) & (test_health < 60.0)),
            ("Near Failure (Health < 40%)", test_health < 40.0),
        ]

        for region_name, mask in regions:
            count = int(np.sum(mask))
            if count > 0:
                sub_y_true = y_test.iloc[mask]
                sub_rf_pred = y_pred_rf[mask]
                sub_xgb_pred = y_pred_xgb[mask]

                rf_reg_mae = round(float(mean_absolute_error(sub_y_true, sub_rf_pred)), 2)
                rf_reg_rmse = round(float(np.sqrt(mean_squared_error(sub_y_true, sub_rf_pred))), 2)

                xgb_reg_mae = round(float(mean_absolute_error(sub_y_true, sub_xgb_pred)), 2)
                xgb_reg_rmse = round(float(np.sqrt(mean_squared_error(sub_y_true, sub_xgb_pred))), 2)

                rf_reg_r2 = round(float(r2_score(sub_y_true, sub_rf_pred)), 4) if count > 10 and sub_y_true.nunique() > 1 else None
                xgb_reg_r2 = round(float(r2_score(sub_y_true, sub_xgb_pred)), 4) if count > 10 and sub_y_true.nunique() > 1 else None

                lifecycle_results.append({
                    "region": region_name,
                    "sample_count": count,
                    "random_forest": {"r2": rf_reg_r2, "mae": rf_reg_mae, "rmse": rf_reg_rmse},
                    "xgboost": {"r2": xgb_reg_r2, "mae": xgb_reg_mae, "rmse": xgb_reg_rmse},
                    "mae_delta": round(float(xgb_reg_mae - rf_reg_mae), 2)
                })

    # Save Results JSON for API consumption
    results_json = {
        "experiment_name": "Controlled 10-Feature XGBoost Regressor Experiment (v2)",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "production_model": "Random Forest Regressor (10 Features)",
        "feature_count": 10,
        "features": FEATURE_NAMES,
        "dataset_size": len(df),
        "train_size": len(X_train_scaled),
        "test_size": len(X_test_scaled),
        "baseline_3feature": baseline_3feat_metrics,
        "random_forest": {
            "r2_score": round(rf_r2, 4),
            "mae": round(rf_mae, 2),
            "rmse": round(rf_rmse, 2),
            "feature_importance": [
                {"feature": name, "importance": imp} for name, imp in zip(FEATURE_NAMES, rf_importances)
            ]
        },
        "xgboost": {
            "algorithm": "XGBRegressor",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "r2_score": round(xgb_r2, 4),
            "mae": round(xgb_mae, 2),
            "rmse": round(xgb_rmse, 2),
            "feature_importance": [
                {"feature": name, "importance": imp} for name, imp in zip(FEATURE_NAMES, xgb_importances)
            ]
        },
        "deltas": {
            "r2_score_delta": delta_r2,
            "mae_days_delta": delta_mae,
            "rmse_days_delta": delta_rmse
        },
        "lifecycle_variable": lifecycle_col,
        "lifecycle_evaluation": lifecycle_results
    }

    json_path = os.path.join(MODELS_DIR, "xgboost_experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n[OK] Experiment results successfully saved to: {json_path}")
    print("=" * 55)

    return results_json

if __name__ == "__main__":
    run_xgboost_experiment()
