"""
Script to train and persist Isolation Forest models for:
1. Fleet-wide baseline (using datasets/sensor_data_10sensors.csv)
2. Per-machine models for all 8 machines (M-001 through M-008) generated via FleetSimulator
"""

import os
import sys
import pandas as pd

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    p = os.path.join(base_dir, folder)
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from isolation_forest import FleetAnomalyDetector, SENSOR_FEATURES
from simulator import FleetSimulator

def train_and_save_anomaly_models(contamination=0.05):
    print("=" * 60)
    print(" TRAINING UNSUPERVISED ISOLATION FOREST ANOMALY MODELS")
    print(f" Contamination Rate: {contamination} (5% industrial anomaly budget)")
    print("=" * 60)

    # 1. Generate multi-step baseline history for all 8 machines from FleetSimulator
    print("\n[1/3] Generating multi-cycle baseline operational data for 8-machine fleet...")
    sim = FleetSimulator()
    fleet_records = []
    # Collect 150 synchronized steps for the fleet (150 * 8 = 1200 records)
    for _ in range(150):
        step_records = sim.step()
        fleet_records.extend(step_records)
    
    fleet_df = pd.DataFrame(fleet_records)
    print(f"Generated {len(fleet_df)} fleet records across machines: {fleet_df['Machine_ID'].unique().tolist()}")

    # 2. Also incorporate fleet_data_v2.csv for broader general operational coverage
    from config import DATASET_PATH, MODELS_DIR
    csv_path = DATASET_PATH
    if os.path.exists(csv_path):
        print(f"\n[2/3] Incorporating general 10-sensor dataset ({csv_path})...")
        hist_df = pd.read_csv(csv_path)
        # Sample healthy/steady-state records for baseline
        if "Machine_Health" in hist_df.columns:
            steady_df = hist_df[hist_df["Machine_Health"] >= 75.0]
        else:
            steady_df = hist_df
        combined_df = pd.concat([fleet_df, steady_df.sample(n=min(5000, len(steady_df)), random_state=42)], ignore_index=True)
    else:
        combined_df = fleet_df

    # 3. Train FleetAnomalyDetector
    print(f"\n[3/3] Training FleetAnomalyDetector on {len(combined_df)} samples...")
    detector = FleetAnomalyDetector(contamination=contamination, n_estimators=150, random_state=42)
    detector.fit_fleet(combined_df, machine_col="Machine_ID")

    # Save models
    models_dir = MODELS_DIR
    detector.save(models_dir)
    print(f"\nSaved models to {models_dir}:")
    for f in os.listdir(models_dir):
        if "isolation_forest" in f:
            print(f"  [OK] {f}")

    # 4. Quick inference sanity check
    print("\nSanity Check on M-001 vs M-008:")
    m1_sample = fleet_df[fleet_df["Machine_ID"] == "M-001"].iloc[-1].to_dict()
    m8_sample = fleet_df[fleet_df["Machine_ID"] == "M-008"].iloc[-1].to_dict()

    m1_eval = detector.predict_reading(m1_sample, machine_id="M-001")
    m8_eval = detector.predict_reading(m8_sample, machine_id="M-008")

    print(f"  M-001 (Healthy Machine) -> Label: {m1_eval['anomaly_label']}, Score: {m1_eval['anomaly_score']}, Severity: {m1_eval['anomaly_severity']}")
    print(f"  M-008 (Worn Machine)    -> Label: {m8_eval['anomaly_label']}, Score: {m8_eval['anomaly_score']}, Severity: {m8_eval['anomaly_severity']}")
    print("=" * 60)
    print("Training and validation complete!")

if __name__ == "__main__":
    train_and_save_anomaly_models()
