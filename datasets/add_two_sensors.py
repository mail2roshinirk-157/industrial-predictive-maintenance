"""
Adds 2 new physics-based sensor columns to the existing turbine motor dataset:

1. Torque (Nm)
   Real mechanical physics formula: T = 9550 * P(kW) / RPM
   Derived from the EXISTING Power_Consumption and RPM columns, plus a small
   sensor measurement noise term (so it isn't a perfectly deterministic
   duplicate of two other columns).

2. Acoustic_Emission (dB)
   AE sensors measure high-frequency stress waves from micro-cracks/bearing
   faults. Physically, dB is a LOG scale of signal energy, so it is modeled
   as a log relationship to Vibration (log-scale, same physics-y approach as
   your other simulated sensors) + measurement noise.

Both formulas were calibrated against the real value ranges in your uploaded
sensor_data.csv (healthy baseline vs near-failure values), so the new columns
sit naturally alongside your existing 8 sensors.

Usage:
    python add_two_sensors.py sensor_data.csv sensor_data_10sensors.csv
"""

import sys
import numpy as np
import pandas as pd


def add_new_sensors(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)

    # --- 1. Torque (Nm) : T = 9550 * Power(kW) / RPM ------------------------
    torque_calc = 9550 * df["Power_Consumption"] / df["RPM"]
    torque_noise = (
        rng.normal(0, 0.02, n) * torque_calc      # ~2% proportional noise
        + rng.normal(0, 0.4, n)                    # small fixed sensor noise
    )
    df["Torque"] = (torque_calc + torque_noise).round(2)

    # --- 2. Acoustic_Emission (dB) : log-scale relation to Vibration --------
    healthy_vibration_baseline = 0.18   # matches this dataset's healthy mean
    ae_calc = 35 + 15 * np.log10(1 + df["Vibration"] / healthy_vibration_baseline)
    ae_noise = rng.normal(0, 1.0, n)
    df["Acoustic_Emission"] = (ae_calc + ae_noise).round(2)

    # --- reorder columns: insert new sensors after Power_Consumption --------
    cols = list(df.columns)
    cols.remove("Torque")
    cols.remove("Acoustic_Emission")
    insert_at = cols.index("Power_Consumption") + 1
    cols[insert_at:insert_at] = ["Torque", "Acoustic_Emission"]
    return df[cols]


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "sensor_data.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sensor_data_10sensors.csv"

    df = pd.read_csv(in_path)
    df = add_new_sensors(df)
    df.to_csv(out_path, index=False)

    print(f"Added Torque + Acoustic_Emission -> {out_path}")
    print(df[["Torque", "Acoustic_Emission"]].describe())
