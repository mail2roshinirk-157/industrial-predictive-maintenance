"""
feature_check.py
================
Diagnostic script to assess whether Torque and Acoustic_Emission
are redundant, noise, or genuinely useful for predicting RUL.

Does NOT modify any existing project files.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=" * 80)
    print(" FEATURE EVALUATION: TORQUE & ACOUSTIC EMISSION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Load Data & Establish Exact Split
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "datasets", "sensor_data_10sensors.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "data", "sensor_data_10sensors.csv")
    
    print(f"\n[Step 0] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df.ffill(inplace=True)
    print(f"Total dataset shape: {df.shape}")

    all_10_features = [
        "Temperature",
        "Vibration",
        "Motor_Current",
        "Pressure",
        "RPM",
        "Flow_Rate",
        "Oil_Temperature",
        "Power_Consumption",
        "Torque",
        "Acoustic_Emission"
    ]
    target_col = "Remaining_Useful_Life_Days"

    # -------------------------------------------------------------------------
    # 1. Correlations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 1. CORRELATION ANALYSIS")
    print("=" * 80)
    corr_matrix = df[all_10_features + [target_col]].corr()

    corr_torque_rul = corr_matrix.loc["Torque", target_col]
    corr_torque_curr = corr_matrix.loc["Torque", "Motor_Current"]
    corr_torque_power = corr_matrix.loc["Torque", "Power_Consumption"]
    corr_torque_rpm = corr_matrix.loc["Torque", "RPM"]

    corr_ae_rul = corr_matrix.loc["Acoustic_Emission", target_col]
    corr_ae_curr = corr_matrix.loc["Acoustic_Emission", "Motor_Current"]
    corr_ae_vib = corr_matrix.loc["Acoustic_Emission", "Vibration"]

    print(f"Torque correlations:")
    print(f"  • with Remaining_Useful_Life_Days : {corr_torque_rul:+.4f}")
    print(f"  • with Motor_Current             : {corr_torque_curr:+.4f}")
    print(f"  • with Power_Consumption        : {corr_torque_power:+.4f} (Underlying driver)")
    print(f"  • with RPM                      : {corr_torque_rpm:+.4f} (Underlying driver)")

    print(f"\nAcoustic_Emission correlations:")
    print(f"  • with Remaining_Useful_Life_Days : {corr_ae_rul:+.4f}")
    print(f"  • with Motor_Current             : {corr_ae_curr:+.4f}")
    print(f"  • with Vibration                 : {corr_ae_vib:+.4f} (Underlying driver)")

    # -------------------------------------------------------------------------
    # Train / Test Split (identical to preprocessing.py: test_size=0.2, random_state=42)
    # -------------------------------------------------------------------------
    X_10 = df[all_10_features]
    y = df[target_col]

    X_train_10, X_test_10, y_train, y_test = train_test_split(
        X_10, y, test_size=0.2, random_state=42
    )

    scaler_10 = StandardScaler()
    X_train_10_scaled = scaler_10.fit_transform(X_train_10)
    X_test_10_scaled = scaler_10.transform(X_test_10)

    # -------------------------------------------------------------------------
    # 3. Model Training & Comparison: (a) 10 features vs (b) 8 features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. MODEL TRAINING & PERFORMANCE COMPARISON")
    print("    Hyperparameters: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)")
    print("=" * 80)

    # (a) Model with all 10 features
    rf_model_path = os.path.join(base_dir, "models", "rf_rul_model.pkl")
    rf_10 = None
    if os.path.exists(rf_model_path):
        try:
            print("\nAttempting to load existing production 10-feature RF model...")
            loaded_m = joblib.load(rf_model_path)
            if getattr(loaded_m, "n_features_in_", 0) == 10:
                rf_10 = loaded_m
                print("Loaded existing models/rf_rul_model.pkl successfully.")
        except Exception as ex:
            print(f"Could not load pre-existing model: {ex}")
            rf_10 = None

    if rf_10 is None:
        print("Training Random Forest on (a) all 10 features...")
        rf_10 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_10.fit(X_train_10_scaled, y_train)

    y_pred_10 = rf_10.predict(X_test_10_scaled)
    mae_10 = mean_absolute_error(y_test, y_pred_10)
    rmse_10 = np.sqrt(mean_squared_error(y_test, y_pred_10))
    r2_10 = r2_score(y_test, y_pred_10)

    # (b) Model without Torque and Acoustic_Emission (8 features)
    features_8 = [
        "Temperature",
        "Vibration",
        "Motor_Current",
        "Pressure",
        "RPM",
        "Flow_Rate",
        "Oil_Temperature",
        "Power_Consumption"
    ]
    print("\nTraining Random Forest on (b) 8 features (WITHOUT Torque & Acoustic_Emission)...")
    X_train_8 = X_train_10[features_8]
    X_test_8 = X_test_10[features_8]

    scaler_8 = StandardScaler()
    X_train_8_scaled = scaler_8.fit_transform(X_train_8)
    X_test_8_scaled = scaler_8.transform(X_test_8)

    rf_8 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_8.fit(X_train_8_scaled, y_train)

    y_pred_8 = rf_8.predict(X_test_8_scaled)
    mae_8 = mean_absolute_error(y_test, y_pred_8)
    rmse_8 = np.sqrt(mean_squared_error(y_test, y_pred_8))
    r2_8 = r2_score(y_test, y_pred_8)

    print("\n" + "-" * 80)
    print(f"{'Metric':<25} | {'(a) All 10 Features':<22} | {'(b) Without Torque & AE':<24} | {'Difference (b - a)':<18}")
    print("-" * 80)
    print(f"{'MAE (Days)':<25} | {mae_10:<22.4f} | {mae_8:<24.4f} | {mae_8 - mae_10:+18.4f}")
    print(f"{'RMSE (Days)':<25} | {rmse_10:<22.4f} | {rmse_8:<24.4f} | {rmse_8 - rmse_10:+18.4f}")
    print(f"{'R² Score':<25} | {r2_10:<22.6f} | {r2_8:<24.6f} | {r2_8 - r2_10:+18.6f}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 2. Permutation Importance for all 10 features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 2. PERMUTATION IMPORTANCE (TEST SET)")
    print("    Computing 5 permutation repeats across 18,250 test samples...")
    print("=" * 80)
    perm_results = permutation_importance(
        rf_10,
        X_test_10_scaled,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    perm_df = pd.DataFrame({
        "Feature": all_10_features,
        "Permutation_Importance_Mean": perm_results.importances_mean,
        "Permutation_Importance_Std": perm_results.importances_std,
        "Gini_Importance": rf_10.feature_importances_
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)

    print(f"\n{'Rank':<5} | {'Feature':<20} | {'Permutation Importance (RMSE drop)':<35} | {'MDI (Gini)':<12}")
    print("-" * 80)
    for idx, row in perm_df.iterrows():
        print(f"{idx+1:<5} | {row['Feature']:<20} | {row['Permutation_Importance_Mean']:+.4f} ± {row['Permutation_Importance_Std']:.4f}             | {row['Gini_Importance']:.4f}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 4. Generation Formulas & Details from Data Generation Script
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 4. HOW TORQUE & ACOUSTIC EMISSION WERE GENERATED")
    print("=" * 80)
    print("""
From 'datasets/add_two_sensors.py':

[A] Torque (Nm):
    Formula:
        torque_calc = 9550 * Power_Consumption / RPM
        torque_noise = rng.normal(0, 0.02, n) * torque_calc + rng.normal(0, 0.4, n)
        Torque = round(torque_calc + torque_noise, 2)

    Dependency on Health / RUL:
        - Torque is NOT generated directly from Health or RUL.
        - Instead, it is a deterministic physical function of Power_Consumption and RPM.
        - In the simulator:
            * Power_Consumption is derived from Motor_Current (raw_curr * 1.85) and increases with wear.
            * RPM decreases with wear (-55 * p^1.4).
        - Therefore, Torque inherits a strong indirect monotonic correlation with degradation.
    Noise Level:
        - Proportional noise: Gaussian ~2% (std = 0.02).
        - Fixed sensor noise: Gaussian ~0.4 Nm (std = 0.4).

[B] Acoustic_Emission (dB):
    Formula:
        healthy_vibration_baseline = 0.18
        ae_calc = 35 + 15 * log10(1 + Vibration / healthy_vibration_baseline)
        ae_noise = rng.normal(0, 1.0, n)
        Acoustic_Emission = round(ae_calc + ae_noise, 2)

    Dependency on Health / RUL:
        - Acoustic_Emission is NOT generated directly from Health or RUL.
        - It is modeled as a direct logarithmic transfer function of Vibration.
        - In the simulator:
            * Vibration increases with degradation (+4.3 * p^1.7) and experiences overload peaks.
        - Therefore, Acoustic_Emission tracks Vibration in log scale.
    Noise Level:
        - Fixed sensor noise: Gaussian std = 1.0 dB.
    """)

    print("=" * 80)
    print(" Feature evaluation complete.")
    print("=" * 80)

if __name__ == "__main__":
    main()
