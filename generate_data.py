"""
Generates simulated sensor data for 20 industrial machines.
Each machine slowly degrades over time: temperature and vibration rise,
pressure and rpm fall, current draw increases.
"""
import os

import numpy as np
import pandas as pd

# create the data folder automatically if it doesn't exist
os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)

NUM_MACHINES = 20
READINGS_PER_MACHINE = 500
MACHINE_TYPES = ["Motor", "Pump", "Compressor", "Turbine"]

rows = []
for machine_id in range(1, NUM_MACHINES + 1):
    machine_type = MACHINE_TYPES[machine_id % len(MACHINE_TYPES)]
    wear_speed = rng.uniform(0.7, 1.3)  # some machines wear faster

    for t in range(READINGS_PER_MACHINE):
        # degradation grows from 0 (new) to ~1 (worn out), accelerating over time
        degradation = min(1.0, (t / READINGS_PER_MACHINE) ** 2 * wear_speed)

        rows.append({
            "machine_id": machine_id,
            "machine_type": machine_type,
            "time_step": t,
            "temperature": 60 + 25 * degradation + rng.normal(0, 2),
            "vibration": 2 + 6 * degradation + rng.normal(0, 0.3),
            "pressure": 100 - 20 * degradation + rng.normal(0, 2),
            "rpm": 1500 - 150 * degradation + rng.normal(0, 20),
            "current": 10 + 4 * degradation + rng.normal(0, 0.3),
            # label for ML: 1 = machine is in the failure-risk zone
            "failure_risk": int(degradation > 0.6),
        })

df = pd.DataFrame(rows)
df.to_csv("data/sensor_data.csv", index=False)

print(f"Saved {len(df)} rows to data/sensor_data.csv")
print(df.head())
print("\nFailure-risk share:", round(df["failure_risk"].mean() * 100, 1), "%")