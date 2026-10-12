import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Single Source of Truth for Dataset and Model Paths
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "fleet_data_v2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Canonical 10-Feature Schema (v2)
FEATURE_COLUMNS = [
    "Temperature",
    "Vibration",
    "Motor_Current",
    "Pressure",
    "RPM",
    "Flow_Rate",
    "Oil_Temperature",
    "Power_Consumption",
    "Load_Percent",
    "Oil_Debris_Count"
]

# Baseline 3-Feature Schema
BASELINE_3FEATURE_COLUMNS = [
    "Temperature",
    "Vibration",
    "Motor_Current"
]

TARGET_COLUMN = "Remaining_Useful_Life_Days"

# Non-feature metadata and label columns excluded from ML models
EXCLUDED_COLUMNS = [
    "Timestamp",
    "Machine_ID",
    "Machine_Name",
    "Day",
    "Tick",
    "Active_Event",
    "Remaining_Useful_Life_Days",
    "Machine_Health",
    "Machine_Status",
    "Predicted_RUL",
    "Raw_Predicted_RUL",
    "Recommended_Action",
    "Inspection_Priority",
    "Next_Inspection_Window",
    "Abnormal_Indicators"
]

# Data Source Configuration ("direct" or "mqtt")
DATA_SOURCE = os.getenv("DATA_SOURCE", "direct").lower()

# MQTT Configuration (Optional single-machine telemetry ingestion)
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "industrial/telemetry/M-001")
MQTT_MACHINE_ID = os.getenv("MQTT_MACHINE_ID", "M-001")
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))

