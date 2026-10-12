"""
ablation_v2.py
==============
Runs systematic ablation study across 5 configurations using RandomForestRegressor:
(a) Baseline on ORIGINAL dataset with 10 features (includes Torque, Acoustic_Emission)
(b) 8 remaining original sensors on NEW fleet_data_v2.csv
(c) 8 sensors + Load_Percent + Oil_Debris_Count (10 features) on fleet_data_v2.csv
(d) 8 sensors + Load_Percent only (9 features) on fleet_data_v2.csv
(e) 8 sensors + Oil_Debris_Count only (9 features) on fleet_data_v2.csv

Evaluates test MAE, RMSE, and R2 on identical train/test splits (test_size=0.2, random_state=42).
Computes permutation importance on test set for model (c).
Saves all printed outputs to datasets/ablation_v2_results.txt.
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

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, "datasets", "sensor_data_10sensors.csv")
    new_path = os.path.join(base_dir, "datasets", "fleet_data_v2.csv")
    out_txt_path = os.path.join(base_dir, "datasets", "ablation_v2_results.txt")

    sys.stdout = DualLogger(out_txt_path)

    print("=" * 85)
    print(" PREDICTIVE MAINTENANCE FEATURE ABLATION STUDY (V2)")
    print("=" * 85)
    print(f"Original dataset path: {orig_path}")
    print(f"New dataset path:      {new_path}")
    print(f"Results log:           {out_txt_path}")

    # Load datasets
    df_orig = pd.read_csv(orig_path)
    df_orig.ffill(inplace=True)

    df_new = pd.read_csv(new_path)
    df_new.ffill(inplace=True)

    target_col = "Remaining_Useful_Life_Days"
    n_samples = len(df_orig)

    # Identical train / test indices
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.2, random_state=42
    )

    y_train = df_orig[target_col].iloc[train_idx].values
    y_test = df_orig[target_col].iloc[test_idx].values

    # Feature definitions
    orig_10_features = [
        "Temperature", "Vibration", "Motor_Current", "Pressure", "RPM",
        "Flow_Rate", "Oil_Temperature", "Power_Consumption", "Torque", "Acoustic_Emission"
    ]
    features_8 = [
        "Temperature", "Vibration", "Motor_Current", "Pressure", "RPM",
        "Flow_Rate", "Oil_Temperature", "Power_Consumption"
    ]
    features_10_new = features_8 + ["Load_Percent", "Oil_Debris_Count"]
    features_load_only = features_8 + ["Load_Percent"]
    features_debris_only = features_8 + ["Oil_Debris_Count"]

    models_config = [
        ("(a) Baseline 10 features (ORIGINAL dataset)", df_orig, orig_10_features),
        ("(b) 8 original sensors (NEW dataset)", df_new, features_8),
        ("(c) 8 sensors + Load_Percent + Oil_Debris_Count", df_new, features_10_new),
        ("(d) 8 sensors + Load_Percent only", df_new, features_load_only),
        ("(e) 8 sensors + Oil_Debris_Count only", df_new, features_debris_only),
    ]

    results = []
    trained_models = {}

    print("\n" + "-" * 85)
    print(" TRAINING & EVALUATING MODELS")
    print(" Hyperparameters: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)")
    print("-" * 85)

    for label, df_data, feat_list in models_config:
        print(f"\nTraining {label} ({len(feat_list)} features)...")
        X_tr = df_data[feat_list].iloc[train_idx].values
        X_te = df_data[feat_list].iloc[test_idx].values

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr_scaled, y_train)

        y_pred = rf.predict(X_te_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Configuration": label,
            "Num_Features": len(feat_list),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        trained_models[label] = (rf, scaler, feat_list, X_te_scaled)
        print(f"  -> MAE: {mae:.4f} Days | RMSE: {rmse:.4f} Days | R²: {r2:.6f}")

    # Summary table
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print(" ABLATION STUDY RESULTS SUMMARY")
    print("=" * 85)
    print(f"{'Model Configuration':<48} | {'Feats':<5} | {'MAE (Days)':<10} | {'RMSE (Days)':<11} | {'R² Score':<9}")
    print("-" * 92)
    for _, row in res_df.iterrows():
        print(f"{row['Configuration']:<48} | {row['Num_Features']:<5} | {row['MAE']:<10.4f} | {row['RMSE']:<11.4f} | {row['R2']:<9.6f}")
    print("-" * 92)

    # Performance Deltas relative to Baseline (a) and 8-sensor (b)
    base_mae = res_df.loc[res_df["Configuration"].str.startswith("(a)"), "MAE"].values[0]
    m8_mae = res_df.loc[res_df["Configuration"].str.startswith("(b)"), "MAE"].values[0]
    print(f"\nDelta vs Baseline (a) [MAE = {base_mae:.4f} Days]:")
    for _, row in res_df.iterrows():
        print(f"  {row['Configuration']:<48}: Delta MAE = {row['MAE'] - base_mae:+.4f} Days")

    print(f"\nDelta vs 8-sensor (b) [MAE = {m8_mae:.4f} Days]:")
    for _, row in res_df.iterrows():
        print(f"  {row['Configuration']:<48}: Delta MAE = {row['MAE'] - m8_mae:+.4f} Days")

    # Permutation Importance for Model (c)
    print("\n" + "=" * 85)
    print(" PERMUTATION IMPORTANCE ON TEST SET: MODEL (c)")
    print(" (8 Sensors + Load_Percent + Oil_Debris_Count)")
    print(" Scoring metric: neg_root_mean_squared_error, 5 repeats, random_state=42")
    print("=" * 85)

    rf_c, scaler_c, feats_c, X_te_scaled_c = trained_models["(c) 8 sensors + Load_Percent + Oil_Debris_Count"]

    perm = permutation_importance(
        rf_c,
        X_te_scaled_c,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    perm_df = pd.DataFrame({
        "Feature": feats_c,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
        "Gini_Importance": rf_c.feature_importances_
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)

    print(f"{'Rank':<5} | {'Feature':<22} | {'Permutation Importance (RMSE drop)':<35} | {'MDI (Gini)':<10}")
    print("-" * 80)
    for idx, row in perm_df.iterrows():
        print(f"{idx+1:<5} | {row['Feature']:<22} | {row['Permutation_Importance_Mean']:>+10.4f} ± {row['Permutation_Importance_Std']:.4f}                | {row['Gini_Importance']:.4f}")
    print("-" * 80)
    print("\nAblation study completed successfully. Results saved to datasets/ablation_v2_results.txt.")

if __name__ == "__main__":
    main()
