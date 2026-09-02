# Industrial Predictive Maintenance Platform - Complete Project Overview

Welcome to the **Industrial Predictive Maintenance Platform**! This document is a complete, beginner-friendly guide explaining what this project is, why it was built, the tech stack behind it, and how every component works under the hood.

---

## 📌 1. Executive Summary

In modern industrial facilities (manufacturing plants, power stations, wind farms), equipment failure leads to costly unplanned downtime, safety hazards, and expensive emergency repairs.

This project is a **Digital Twin SCADA & Machine Learning Platform** that monitors an industrial asset (e.g., Turbine Motor Unit A1) in real-time. It predicts the **Remaining Useful Life (RUL)** in days before failure occurs, evaluates machine health, and automatically issues intelligent servicing recommendations to maintenance engineers before a catastrophic breakdown happens.

---

## 🛠️ 2. Tech Stack Overview

The project is built using a modern, decoupled full-stack architecture combining Data Science, Machine Learning, Web Backend Services, and an Interactive SCADA Frontend.

| Layer | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Machine Learning** | Python 3.10+, Scikit-learn, Pandas, NumPy, Joblib | Model training, data preprocessing, feature engineering, and RUL estimation |
| **Backend API** | FastAPI, Uvicorn, Asyncio, Pydantic | Asynchronous REST server, real-time simulation clock loop, and single-source-of-truth state management |
| **Frontend UI** | React.js (Vite), TailwindCSS, Plotly.js (`react-plotly.js`), Lucide React | High-performance interactive dashboard, real-time live sensor streaming, fixed engineering chart scaling, and maintenance decision support |
| **Version Control** | Git, GitHub | Source control and project distribution |

---

## 🏗️ 3. System Architecture & Project Structure

```text
predictive_maintenance/
│
├── main.py                     # FastAPI backend application & simulation clock loop
├── requirements.txt            # Python dependencies (fastapi, uvicorn, scikit-learn, pandas, etc.)
│
├── simulator/                  # Digital Twin Physical Simulation Engine
│   └── simulator.py            # Physics-based SCADA telemetry generator & monotonic health model
│
├── model/                      # Machine Learning & Decision Engine
│   ├── predict.py              # RF model loader, RUL temporal smoothing filter, & future trend extrapolation
│   ├── maintenance_engine.py   # Single-source-of-truth industrial decision policy engine
│   └── train_model.py          # Random Forest Regressor training script
│
├── preprocessing/              # Data Pipeline
│   └── preprocessing.py        # StandardScaler & feature preprocessing utilities
│
├── datasets/                   # Synthetic Dataset Generator
│   ├── dataset_generator.py    # Generates 250-machine 365-day training dataset
│   └── sensor_data.csv         # Generated SCADA training data
│
├── utils/                      # Helper Utilities
│   └── utils.py                # Status color tokens & markdown formatting helpers
│
└── frontend/                   # React + Vite Frontend Web Application
    ├── index.html              # Main HTML entry point
    ├── vite.config.js          # Vite build & dev server configuration
    ├── package.json            # Node.js dependencies (React, TailwindCSS, Plotly, Lucide)
    └── src/
        ├── App.jsx             # Main React app layout & navigation router
        ├── pages/
        │   ├── Dashboard.jsx   # Real-time SCADA monitoring dashboard
        │   ├── Maintenance.jsx # Engineer maintenance decision support page
        │   ├── Analytics.jsx   # Multi-sensor lifecycle analytics page
        │   └── Evaluation.jsx  # ML model accuracy & feature importance page
        ├── components/
        │   ├── LiveChart.jsx   # Streaming Plotly sensor graph with fixed engineering Y-ranges
        │   ├── MaintenanceDashboard.jsx # Decision cards & full-width RUL projection chart
        │   ├── GaugeCard.jsx   # Dynamic radial health gauge
        │   ├── MetricCard.jsx  # Industrial KPI metric card
        │   ├── StatusBadge.jsx # Color-coded status badge component
        │   └── SensorTable.jsx # Recent 10 sensor log table
        └── services/
            └── api.js          # Frontend HTTP client connecting to FastAPI backend
```

---

## 🔄 4. How Everything Works (Step-by-Step Execution Flow)

### Step 1: Synthetic SCADA Dataset Generation
- **Script:** `datasets/dataset_generator.py`
- **What it does:** Simulates **250 unique industrial turbine machines** operating across a **365-day lifespan** (~91,250 records).
- **Data Attributes:**
  - `Timestamp`, `Machine_ID`, `Temperature (°C)`, `Vibration RMS (mm/s)`, `Motor Current (A)`, `Acoustic Noise (dB)`, `Machine Health (%)`, and `Remaining Useful Life (Days)`.

### Step 2: Machine Learning Model Training
- **Script:** `model/train_model.py` & `preprocessing/preprocessing.py`
- **Algorithm:** **Random Forest Regressor** (100 Decision Trees).
- **Process:**
  1. Loads `sensor_data.csv`.
  2. Scales features (`Temperature`, `Vibration`, `Motor_Current`) using `StandardScaler`.
  3. Trains the Random Forest model to predict `Remaining Useful Life`.
  4. Computes model evaluation metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score (~0.98+)**.
  5. Saves trained artifacts to disk (`models/rf_rul_model.pkl` and `models/scaler.pkl`).

### Step 3: Real-Time Digital Twin Telemetry Simulation
- **Script:** `simulator/simulator.py`
- **Concept:** Acts as a live SCADA hardware simulator generating realistic sensor readings tick-by-tick across 4 machine lifecycle phases:
  1. **Phase 1 (Healthy Operation, Days 1–150):** Low-frequency micro-drift + stationary Gaussian noise + diurnal sinusoidal cycles.
  2. **Phase 2 (Early Wear, Days 151–260):** Smooth continuous wear onset (sigmoid blending) with mean rise and heteroskedastic variance growth ($\sigma \propto \text{wear}^{1.5}$).
  3. **Phase 3 (Progressive Wear, Days 261–330):** Accelerated degradation, random thermal bursts, overload current spikes, and vibration oscillations.
  4. **Phase 4 (Critical Stage, Days 331–365+):** Machine reaches terminal state.
- **Physics Rules:**
  - **Correlated Multi-Sensor Coupling:** Temperature increases drive motor current demand ($r = 0.99$), and mechanical vibration amplifies acoustic noise ($r = 0.991$).
  - **Monotonic Health Accumulation:** Machine health strictly decreases (`health = min(prev_health, computed_health)`) and clamps permanently at `0.0%` upon failure until reset.
  - **Living Post-Failure Telemetry:** Sensors continue producing active telemetry after failure because real industrial equipment continues generating signals until manually shut down.

### Step 4: RUL Prediction Filtering & Stabilization
- **Script:** `model/predict.py`
- **Problem solved:** Raw machine learning predictions can fluctuate day-to-day due to signal noise (e.g., predicting 263 days, then 265 days, then 261 days).
- **Solution (`filter_displayed_rul`):**
  - Applies a temporal smoothing filter that blends raw ML predictions with a time-decay target.
  - Enforces a strict **monotonic non-increasing bound** (`displayed_RUL <= prev_displayed_RUL`).
  - Prevents multi-step integer plateaus (`263 -> 262 -> 261 -> 260...`).
  - Clamps displayed RUL to `0` when Machine Health reaches `0%`.
- **Trend Extrapolation (`get_future_trend`):** Projects a 20-step future trend starting seamlessly from the latest historical point with an expanding uncertainty fan.

### Step 5: Single-Source-of-Truth Decision Policy Engine
- **Script:** `model/maintenance_engine.py`
- **What it does:** Serves as the single authority converting Health (%), Displayed RUL (Days), and Active Events into unified operational states:
  - **Status Hierarchy:** `Healthy` $\rightarrow$ `Slight Wear` $\rightarrow$ `Moderate Wear` $\rightarrow$ `Warning` $\rightarrow$ `Critical`.
  - **Monotonic Guard:** Status level never regresses during a run.
  - **Actionable Outputs:** Recommended action, servicing priority (`Low`, `Moderate`, `High`, `Urgent`, `Critical`), and inspection window.

### Step 6: Asynchronous Web Server & State Management
- **Script:** `main.py`
- **Framework:** FastAPI running on Uvicorn.
- **Clock Loop (`simulation_clock_loop`):** Runs an asynchronous background task that ticks the simulator engine, computes RUL predictions, updates status, and appends records to history.
- **REST Endpoints:**
  - `GET /api/status`: System status and model metadata.
  - `GET /api/current`: Current telemetry snapshot, Health, Displayed RUL, and Machine Status.
  - `GET /api/history`: Full historical telemetry series + 20-step trend projections.
  - `GET /api/maintenance`: Single-source-of-truth maintenance decision policy payload.
  - `GET /api/model`: Model evaluation metrics (MAE, RMSE, $R^2$, Feature Importance, Residuals).
  - `GET /api/logs`: Latest 10 SCADA log entries.
  - `POST /api/control`: Controls simulation clock (`start`, `pause`, `next`, `reset`, `set_speed`). Automatically auto-resets when starting from a terminal 0% health state.

### Step 7: React SCADA Dashboard UI
- **Folder:** `frontend/src/`
- **Pages & Components:**
  - **`Dashboard.jsx`**: Main SCADA view with radial Health Gauge, RUL KPI card, Machine Stage & Alert badges, Live Plotly chart with fixed engineering Y-ranges (Temp: 55-85°C, Vib: 0-6 mm/s, Current: 5-25 A), and Recent Logs Table.
  - **`Maintenance.jsx`**: Maintenance decision support view with action guidance, servicing priority badges, and a full-width RUL projection chart.
  - **`Analytics.jsx`**: Multi-sensor lifecycle trend charts with condition degradation markers.
  - **`Evaluation.jsx`**: ML performance dashboard displaying Random Forest accuracy, feature importance charts, and actual vs. predicted scatter plots.

---

## ⚡ 5. How to Run the Project Locally

### Prerequisites
- **Python 3.10+** installed.
- **Node.js 18+** & npm installed.

---

### Step A: Start the FastAPI Backend

1. Open PowerShell / Terminal and navigate to the project directory:
   ```powershell
   cd c:\Users\shiva\.gemini\antigravity\scratch\predictive_maintenance
   ```
2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Start the FastAPI backend server:
   ```powershell
   python main.py
   ```
   *The backend server will run at `http://127.0.0.1:8000`.*

---

### Step B: Start the React Frontend

1. Open a **second** terminal window and navigate to the `frontend` folder:
   ```powershell
   cd c:\Users\shiva\.gemini\antigravity\scratch\predictive_maintenance\frontend
   ```
2. Install Node.js dependencies (first time only):
   ```powershell
   npm install
   ```
3. Launch the React development server:
   ```powershell
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:5173`.

---

## 🔗 6. Repository Link

- **GitHub Repository:** [https://github.com/Siva402-ai/industrial-predictive-maintenance](https://github.com/Siva402-ai/industrial-predictive-maintenance)

---

## 💡 Summary for Non-Technical Readers

Think of this project as a **smart digital doctor for heavy industrial machinery**:
1. **Sensors** on the turbine measure temperature, vibration, current, and noise.
2. **Machine Learning** learns what healthy vs. degrading turbine behavior looks like.
3. The **Digital Twin Simulator** imitates how a machine ages over 365 days.
4. The **Web Dashboard** shows live graphs, tells engineers how many days of life the machine has left, and alerts them when it's time to perform preventive maintenance before a sudden breakdown.
