"""
add_load_and_debris.py
======================
Generates datasets/fleet_data_v2.csv from the original dataset:
1. Drops Torque and Acoustic_Emission.
2. Adds Load_Percent as a slow-varying AR(1) process (clipped to [40, 100], mean ~70),
   independent of wear (p) and RUL. Adds its physical coupling to Motor_Current and
   Vibration, and recomputes Power_Consumption using the simulator's linear factor (1.85).
3. Adds Oil_Debris_Count sampled from Poisson(lambda = 5 + 400 * p^3), where
   p = 1 - Machine_Health / 100.
4. Preserves Machine_Health, Machine_Status, Remaining_Useful_Life_Days, timestamps,
   and row order under random_state=42.
5. Evaluates and prints pairwise correlations, flagging any |r| > 0.95 as REDUNDANT.
"""

import os
import sys
import numpy as np
import pandas as pd

def main():
    print("=" * 80)
    print(" GENERATING FLEET DATA V2: ADDING LOAD_PERCENT & OIL_DEBRIS_COUNT")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base_dir, "datasets", "sensor_data_10sensors.csv")
    if not os.path.exists(in_path):
        in_path = os.path.join(base_dir, "datasets", "sensor_data.csv")
    
    out_path = os.path.join(base_dir, "datasets", "fleet_data_v2.csv")
    
    print(f"Reading input dataset from: {in_path}")
    df = pd.read_csv(in_path)
    print(f"Initial shape: {df.shape}")

    # 1. Drop Torque and Acoustic_Emission
    dropped = []
    for col in ["Torque", "Acoustic_Emission"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
    print(f"Dropped columns: {dropped}")

    # Fixed seed for reproducibility
    seed = 42
    rng = np.random.default_rng(seed)

    # 2. Add Load_Percent per machine in time order (250 machines x 365 steps)
    n_rows = len(df)
    records_per_machine = 365
    n_machines = n_rows // records_per_machine

    print(f"\nGenerating Load_Percent for {n_machines} machines x {records_per_machine} steps as AR(1)...")
    loads = []
    for m in range(n_machines):
        val = 70.0 + rng.normal(0, 4.0)
        for t in range(records_per_machine):
            val = 70.0 + 0.92 * (val - 70.0) + rng.normal(0, 3.5)
            loads.append(np.clip(val, 40.0, 100.0))
    
    df["Load_Percent"] = np.round(loads, 2)
    print(f"Load_Percent summary: mean={df['Load_Percent'].mean():.2f}, "
          f"std={df['Load_Percent'].std():.2f}, "
          f"min={df['Load_Percent'].min():.2f}, "
          f"max={df['Load_Percent'].max():.2f}")

    # Load effect calibration:
    # Empirical sensor change between healthy (p=0) and failure (p=1):
    # Motor_Current delta: ~9.60 A (healthy ~8.02 A, near failure ~17.62 A)
    # Vibration delta: ~3.99 mm/s (healthy ~0.20 mm/s, near failure ~4.19 mm/s)
    # Target load effect: 20-30% of wear delta over the max load excursion (100 - 70 = 30)
    # k1 = (0.25 * 9.60) / 30 = 0.08 A/%
    # k2 = (0.25 * 3.99) / 30 = 0.033 (mm/s)/%
    k1 = 0.08
    k2 = 0.033
    print(f"\nSelected Load coupling coefficients:")
    print(f"  k1 = {k1:.4f} (Motor_Current load effect: 30 * {k1:.4f} = {30*k1:.2f} A, which is {30*k1/9.60*100:.1f}% of wear delta)")
    print(f"  k2 = {k2:.4f} (Vibration load effect:    30 * {k2:.4f} = {30*k2:.2f} mm/s, which is {30*k2/3.99*100:.1f}% of wear delta)")

    # Add load effect to Motor_Current and Vibration
    load_delta = df["Load_Percent"] - 70.0
    df["Motor_Current"] = (df["Motor_Current"] + k1 * load_delta).round(2)
    df["Vibration"] = (df["Vibration"] + k2 * load_delta).round(2)

    # Recompute Power_Consumption using simulator's formula: raw_power = raw_curr * 1.85 + ...
    # Hence delta_Power = 1.85 * delta_Current = 1.85 * k1 * (Load_Percent - 70.0)
    df["Power_Consumption"] = (df["Power_Consumption"] + 1.85 * k1 * load_delta).round(2)
    print("Recomputed Power_Consumption consistently from new Motor_Current (scale factor = 1.85).")

    # 3. Add Oil_Debris_Count
    # Derived directly from p = 1.0 - Machine_Health / 100.0
    # lambda = 5 + 400 * p^3, sampled from Poisson(lambda)
    p = np.clip(1.0 - df["Machine_Health"] / 100.0, 0.0, 1.0)
    lam = 5.0 + 400.0 * (p ** 3)
    df["Oil_Debris_Count"] = rng.poisson(lam)
    print(f"\nOil_Debris_Count summary: mean={df['Oil_Debris_Count'].mean():.2f}, "
          f"min={df['Oil_Debris_Count'].min()}, "
          f"max={df['Oil_Debris_Count'].max()}")

    # Reorder columns: insert Load_Percent and Oil_Debris_Count after Power_Consumption
    cols = [c for c in df.columns if c not in ["Load_Percent", "Oil_Debris_Count"]]
    insert_at = cols.index("Power_Consumption") + 1
    cols[insert_at:insert_at] = ["Load_Percent", "Oil_Debris_Count"]
    df = df[cols]

    # Save to fleet_data_v2.csv
    df.to_csv(out_path, index=False)
    print(f"\nSuccessfully wrote new dataset to: {out_path}")
    print(f"Columns in new dataset: {list(df.columns)}")

    # 5. Correlation analysis & Redundancy check
    print("\n" + "=" * 80)
    print(" CORRELATION ANALYSIS (Load_Percent & Oil_Debris_Count)")
    print("=" * 80)
    sensors = [
        "Temperature", "Vibration", "Motor_Current", "Pressure",
        "RPM", "Flow_Rate", "Oil_Temperature", "Power_Consumption"
    ]
    eval_targets = sensors + ["Remaining_Useful_Life_Days"]

    print(f"{'Variable':<28} | {'Load_Percent r':<16} | {'Oil_Debris_Count r':<20} | {'Redundancy (|r| > 0.95)':<25}")
    print("-" * 95)
    for col in eval_targets:
        r_load = df["Load_Percent"].corr(df[col])
        r_deb = df["Oil_Debris_Count"].corr(df[col])
        flag = []
        if abs(r_load) > 0.95:
            flag.append(f"Load_Percent-{col} REDUNDANT")
        if abs(r_deb) > 0.95:
            flag.append(f"Oil_Debris_Count-{col} REDUNDANT")
        flag_str = ", ".join(flag) if flag else "OK (None > 0.95)"
        print(f"{col:<28} | {r_load:>+14.4f} | {r_deb:>+18.4f} | {flag_str}")
    print("-" * 95)

if __name__ == "__main__":
    main()
