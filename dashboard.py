"""
========================================================================================
INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM - STREAMLIT DASHBOARD
8-Machine Fleet Digital Twin with 10 Sensor Telemetry Parameters
Supervised XGBoost RUL Model + Unsupervised Isolation Forest Anomaly Detection
========================================================================================
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root and subpackages are in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    p = os.path.join(base_dir, folder)
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from simulator import FleetSimulator
from isolation_forest import FleetAnomalyDetector, SENSOR_FEATURES

# Configure Streamlit Page
st.set_page_config(
    page_title="Industrial Predictive Maintenance & Anomaly Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Status color mappings
STATUS_COLORS = {
    "Healthy": "#16A34A",
    "Slight Wear": "#2563EB",
    "Moderate Wear": "#F59E0B",
    "Warning": "#D97706",
    "Critical": "#DC2626",
    "Normal": "#16A34A",
    "anomaly": "#DC2626",
    "normal": "#16A34A",
    "Severe": "#991B1B",
    "High": "#DC2626",
    "Moderate": "#D97706"
}

# --------------------------------------------------------------------------------------
# INDEPENDENT MODEL LOADER (XGBoost RUL & Isolation Forest Anomaly Detector)
# Requirement 6: Keep XGBoost RUL and Isolation Forest strictly independent.
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Loads XGBoost RUL model and Isolation Forest detector independently.
    Failure in one model will NOT prevent the other model from functioning.
    """
    models_dir = os.path.join(base_dir, "models")
    
    # 1. XGBoost RUL Regressor + Scaler
    xgb_model = None
    scaler = None
    xgb_error = None
    try:
        xgb_path = os.path.join(models_dir, "xgb_rul_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(xgb_path) and os.path.exists(scaler_path):
            xgb_model = joblib.load(xgb_path)
            scaler = joblib.load(scaler_path)
        else:
            xgb_error = "Model files xgb_rul_model.pkl or scaler.pkl not found."
    except Exception as ex:
        xgb_error = str(ex)

    # 2. Isolation Forest Fleet Anomaly Detector
    anomaly_detector = None
    anomaly_error = None
    try:
        if os.path.exists(models_dir):
            anomaly_detector = FleetAnomalyDetector.load(models_dir)
        else:
            anomaly_error = "Models directory not found."
    except Exception as ex:
        anomaly_error = str(ex)

    return xgb_model, scaler, xgb_error, anomaly_detector, anomaly_error


xgb_model, scaler, xgb_error, anomaly_detector, anomaly_error = load_models()


# --------------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------------------
if "sim_engine" not in st.session_state:
    st.session_state.sim_engine = FleetSimulator(max_lifespan_days=250, degradation_factor=1.8)
    st.session_state.auto_play = False
    st.session_state.sim_speed = 1.0
    st.session_state.history = {cfg["machine_id"]: [] for cfg in FleetSimulator.MACHINE_CONFIGS}
    st.session_state.recent_anomalies = []
    
    # Run 5 initial warm-up ticks to populate initial baseline telemetry
    for _ in range(5):
        step_records = st.session_state.sim_engine.step()
        for rec in step_records:
            m_id = rec["Machine_ID"]
            
            # Predict XGBoost RUL (Independent)
            rul_pred = None
            if xgb_model is not None and scaler is not None:
                try:
                    feat_df = pd.DataFrame([rec])[SENSOR_FEATURES]
                    scaled_feats = scaler.transform(feat_df)
                    rul_pred = max(0, int(round(xgb_model.predict(scaled_feats)[0])))
                except Exception:
                    rul_pred = None
            rec["XGBoost_RUL"] = rul_pred

            # Predict Isolation Forest Anomaly (Independent)
            anom_eval = None
            if anomaly_detector is not None:
                try:
                    anom_eval = anomaly_detector.predict_reading(rec, machine_id=m_id)
                except Exception:
                    anom_eval = None
            
            rec["Anomaly_Label"] = anom_eval.get("anomaly_label", "normal") if anom_eval else "unknown"
            rec["Anomaly_Score"] = anom_eval.get("anomaly_score", 0.0) if anom_eval else 0.0
            rec["Anomaly_Severity"] = anom_eval.get("anomaly_severity", "Normal") if anom_eval else "Normal"
            rec["Is_Anomaly"] = anom_eval.get("is_anomaly", False) if anom_eval else False
            rec["Model_Type"] = anom_eval.get("model_type", "fleet_wide") if anom_eval else "none"

            st.session_state.history[m_id].append(rec)


def advance_simulation_step():
    """Advances all 8 machines in one synchronized simulation tick."""
    sim = st.session_state.sim_engine
    step_records = sim.step()

    for rec in step_records:
        m_id = rec["Machine_ID"]

        # 1. Independent XGBoost RUL Prediction
        rul_pred = None
        if xgb_model is not None and scaler is not None:
            try:
                feat_df = pd.DataFrame([rec])[SENSOR_FEATURES]
                scaled_feats = scaler.transform(feat_df)
                rul_pred = max(0, int(round(xgb_model.predict(scaled_feats)[0])))
            except Exception:
                rul_pred = None
        rec["XGBoost_RUL"] = rul_pred

        # 2. Independent Isolation Forest Anomaly Detection
        anom_eval = None
        if anomaly_detector is not None:
            try:
                anom_eval = anomaly_detector.predict_reading(rec, machine_id=m_id)
            except Exception:
                anom_eval = None

        is_anom = anom_eval.get("is_anomaly", False) if anom_eval else False
        anom_label = anom_eval.get("anomaly_label", "normal") if anom_eval else "normal"
        anom_score = anom_eval.get("anomaly_score", 0.0) if anom_eval else 0.0
        anom_sev = anom_eval.get("anomaly_severity", "Normal") if anom_eval else "Normal"
        model_type = anom_eval.get("model_type", "fleet_wide") if anom_eval else "none"

        rec["Anomaly_Label"] = anom_label
        rec["Anomaly_Score"] = anom_score
        rec["Anomaly_Severity"] = anom_sev
        rec["Is_Anomaly"] = is_anom
        rec["Model_Type"] = model_type

        # Log recent anomalies
        if is_anom:
            st.session_state.recent_anomalies.insert(0, {
                "Timestamp": rec["Timestamp"],
                "Machine_ID": m_id,
                "Machine_Name": rec["Machine_Name"],
                "Anomaly_Score": anom_score,
                "Severity": anom_sev,
                "Temperature": rec["Temperature"],
                "Vibration": rec["Vibration"],
                "Current": rec["Motor_Current"],
                "Pressure": rec["Pressure"],
                "RPM": rec["RPM"]
            })
            # Keep max 50 recent anomalies
            if len(st.session_state.recent_anomalies) > 50:
                st.session_state.recent_anomalies.pop()

        # Update history
        st.session_state.history[m_id].append(rec)
        if len(st.session_state.history[m_id]) > 300:
            st.session_state.history[m_id].pop(0)


# --------------------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL METADATA
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Fleet Control Center")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.session_state.auto_play:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.auto_play = False
            st.rerun()
    else:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.auto_play = True
            st.rerun()

with col_s2:
    if st.button("⏭️ Next Step", use_container_width=True):
        advance_simulation_step()
        st.rerun()

st.session_state.sim_speed = st.sidebar.slider(
    "Clock Tick Speed (seconds)",
    min_value=0.2, max_value=3.0, value=float(st.session_state.sim_speed), step=0.1
)

if st.sidebar.button("🔄 Reset Entire Fleet", use_container_width=True):
    st.session_state.sim_engine.reset()
    st.session_state.history = {cfg["machine_id"]: [] for cfg in FleetSimulator.MACHINE_CONFIGS}
    st.session_state.recent_anomalies = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Machine Selector")
machine_options = [f"{cfg['machine_id']} - {cfg['name']}" for cfg in FleetSimulator.MACHINE_CONFIGS]
selected_option = st.sidebar.selectbox("Select Target Machine:", machine_options, index=0)
selected_m_id = selected_option.split(" - ")[0]

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 AI Engine Architecture")

# XGBoost Status
if xgb_model is not None:
    st.sidebar.success("✅ **XGBoost RUL Model**: Active (10 Features)")
else:
    st.sidebar.error(f"❌ **XGBoost RUL Model**: Unavailable ({xgb_error})")

# Isolation Forest Status
if anomaly_detector is not None:
    num_per_m = len(anomaly_detector.machine_detectors)
    st.sidebar.success(f"✅ **Isolation Forest**: Active ({num_per_m} Per-Machine + Fleet Fallback)")
else:
    st.sidebar.error(f"❌ **Isolation Forest**: Unavailable ({anomaly_error})")

# Contamination Info
st.sidebar.markdown("""
**Contamination Rate:** `0.05` (5%)  
*Chosen based on industrial SCADA standard: machines operate in nominal steady-state 95% of life, leaving 5% anomaly budget to prevent alarm fatigue.*
""")


# --------------------------------------------------------------------------------------
# AUTO-PLAY CLOCK REFRESH
# --------------------------------------------------------------------------------------
if st.session_state.auto_play:
    advance_simulation_step()
    time.sleep(st.session_state.sim_speed)
    st.rerun()


# --------------------------------------------------------------------------------------
# MAIN APPLICATION TABS
# --------------------------------------------------------------------------------------
st.title("🏭 Industrial Predictive Maintenance & Anomaly Detection")
st.markdown("##### 8-Machine Synchronized Fleet Monitoring | 10 Physics-Coupled SCADA Parameters")

tabs = st.tabs([
    "📡 Fleet SCADA Overview",
    "🛡️ Anomaly Detection (Isolation Forest)",
    "🎯 XGBoost RUL Prediction",
    "📊 Multi-Sensor Lifecycle Analytics",
    "🔬 Machine Learning Diagnostics"
])

current_records = {m_id: hist[-1] for m_id, hist in st.session_state.history.items() if hist}
curr_rec = current_records.get(selected_m_id, {})

# ======================================================================================
# TAB 1: 📡 FLEET SCADA OVERVIEW
# ======================================================================================
with tabs[0]:
    st.subheader("🌐 Synchronized Fleet Status (8 Machines)")
    
    # Fleet Table Data
    fleet_rows = []
    for cfg in FleetSimulator.MACHINE_CONFIGS:
        mid = cfg["machine_id"]
        rec = current_records.get(mid, {})
        if rec:
            fleet_rows.append({
                "Machine ID": mid,
                "Machine Name": cfg["name"],
                "Operating Stage": rec.get("Machine_Status", "Healthy"),
                "Machine Health (%)": f"{rec.get('Machine_Health', 100.0)}%",
                "Ground Truth RUL": f"{rec.get('Remaining_Useful_Life_Days', 0)} d",
                "XGBoost RUL (AI)": f"{rec.get('XGBoost_RUL', '--')} d",
                "Anomaly Status": "🔴 ANOMALY" if rec.get("Is_Anomaly") else "🟢 NORMAL",
                "Anomaly Score": f"{rec.get('Anomaly_Score', 0.0):.4f}",
                "Severity": rec.get("Anomaly_Severity", "Normal"),
                "Active Event": rec.get("Active_Event", "None")
            })
    
    st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True)

    st.markdown("---")
    st.subheader(f"🔍 Unit Deep-Dive: {curr_rec.get('Machine_Name', selected_m_id)} ({selected_m_id})")

    # Primary KPIs Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(label="Machine Health", value=f"{curr_rec.get('Machine_Health', 100.0)}%")
    with kpi2:
        st.metric(label="XGBoost Predicted RUL", value=f"{curr_rec.get('XGBoost_RUL', '--')} Days")
    with kpi3:
        st.metric(label="Ground-Truth RUL", value=f"{curr_rec.get('Remaining_Useful_Life_Days', '--')} Days")
    with kpi4:
        anom_color = "#DC2626" if curr_rec.get("Is_Anomaly") else "#16A34A"
        st.markdown(f"**Anomaly Status (Isolation Forest)**<br><span style='background-color:{anom_color}; color:white; padding:4px 10px; border-radius:4px; font-weight:bold;'>{curr_rec.get('Anomaly_Label', 'normal').upper()}</span>", unsafe_allow_html=True)
    with kpi5:
        st.metric(label="Anomaly Score", value=f"{curr_rec.get('Anomaly_Score', 0.0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 10 Sensor Parameters KPI Grid
    st.write("##### 📡 10-Parameter Live Sensor Telemetry")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    with s_col1:
        st.metric("Temperature", f"{curr_rec.get('Temperature', '--')} °C")
        st.metric("Flow Rate", f"{curr_rec.get('Flow_Rate', '--')} L/min")
    with s_col2:
        st.metric("Vibration RMS", f"{curr_rec.get('Vibration', '--')} mm/s")
        st.metric("Oil Temperature", f"{curr_rec.get('Oil_Temperature', '--')} °C")
    with s_col3:
        st.metric("Motor Current", f"{curr_rec.get('Motor_Current', '--')} A")
        st.metric("Power Consumption", f"{curr_rec.get('Power_Consumption', '--')} kW")
    with s_col4:
        st.metric("Pressure", f"{curr_rec.get('Pressure', '--')} bar")
        st.metric("Torque", f"{curr_rec.get('Torque', '--')} Nm")
    with s_col5:
        st.metric("Speed (RPM)", f"{curr_rec.get('RPM', '--')} RPM")
        st.metric("Acoustic Emission", f"{curr_rec.get('Acoustic_Emission', '--')} dB")


# ======================================================================================
# TAB 2: 🛡️ ANOMALY DETECTION (ISOLATION FOREST)
# Requirement 5: Dedicated tab showing current anomaly status per machine,
# an anomaly score trend chart, and highlighting machines with recent anomalies.
# ======================================================================================
with tabs[1]:
    st.subheader("🛡️ Unsupervised Sensor Anomaly Detection (Isolation Forest)")
    
    # Requirement 7: Educational logic explanation callout
    with st.expander("ℹ️ How Isolation Forest Anomaly Detection Works (Mathematical Principle)", expanded=False):
        st.markdown("""
        **Isolation Forest (iForest)** is an unsupervised algorithm designed specifically for anomaly detection:
        
        1. **Isolation via Random Splits:**
           Unlike traditional methods that calculate data density or normal operating boundaries, Isolation Forest explicitly isolates anomalous observations.
           At each step, the algorithm builds an ensemble of *Isolation Trees (iTrees)* by randomly selecting a sensor feature and randomly picking a split value between that feature's minimum and maximum values.
           
        2. **Shorter Path Length Principle:**
           - **Anomalies** (rare, extreme, or uncorrelated sensor spikes) reside in sparse, low-density regions of the 10-dimensional sensor space. Because they are unusual, they are separated near the top of the tree in **very few splits (shorter path length)**.
           - **Normal steady-state readings** reside in dense clusters and require **many subsequent splits (longer path length)** to isolate into an individual leaf.
           
        3. **Continuous Anomaly Scoring ($s(x, n) = 2^{-E[h(x)] / c(n)}$):**
           - Average path length across the forest is normalized against the average path length of an unsuccessful binary search tree search $c(n)$.
           - **Score $\to$ 1.0:** Very short path length $\implies$ Definite Anomaly.
           - **Score $\to$ 0.0:** Long path length $\implies$ Nominal steady-state.
           - **Decision Boundary ($0.50$):** Samples exceeding 0.50 are categorized as anomalies, calibrated to the **5% contamination rate**.
           
        4. **Zero-Label Advantage:**
           Operates completely independently of supervised RUL or failure labels — crucial for production facilities where run-to-failure records are unavailable.
        """)

    # 1. CURRENT ANOMALY STATUS PER MACHINE (Requirement 5)
    st.write("#### 1️⃣ Current Anomaly Status Across All 8 Fleet Machines")
    m_cards = st.columns(4)
    for idx, cfg in enumerate(FleetSimulator.MACHINE_CONFIGS):
        col = m_cards[idx % 4]
        mid = cfg["machine_id"]
        rec = current_records.get(mid, {})
        is_anom = rec.get("Is_Anomaly", False)
        anom_score = rec.get("Anomaly_Score", 0.0)
        sev = rec.get("Anomaly_Severity", "Normal")
        badge_bg = "#DC2626" if is_anom else "#16A34A"
        
        with col:
            st.markdown(f"""
            <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 12px; background-color: #F8FAFC;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#1E293B;">{mid}</strong>
                    <span style="background-color:{badge_bg}; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold;">
                        {rec.get('Anomaly_Label', 'normal').upper()}
                    </span>
                </div>
                <div style="font-size:13px; color:#64748B; margin-top:4px;">{cfg['name']}</div>
                <div style="margin-top:8px; display:flex; justify-content:space-between;">
                    <span style="font-size:13px;">Anomaly Score:</span>
                    <strong style="font-size:13px; color:{'#DC2626' if is_anom else '#16A34A'};">{anom_score:.4f}</strong>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-size:13px;">Severity:</span>
                    <strong style="font-size:13px;">{sev}</strong>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-size:13px;">Model:</span>
                    <span style="font-size:12px; color:#475569;">{rec.get('Model_Type', 'per_machine')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. HIGHLIGHT MACHINES WITH RECENT ANOMALIES (Requirement 5)
    st.write("#### 2️⃣ Recent Fleet Anomalies Log")
    if st.session_state.recent_anomalies:
        recent_df = pd.DataFrame(st.session_state.recent_anomalies[:10])
        st.warning(f"⚠️ **{len(st.session_state.recent_anomalies)} anomalous events** detected across the fleet. Showing latest 10 occurrences:")
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.success("✅ **No Anomalies Detected in Recent Operating Window.** All 8 fleet machines are currently operating within nominal multi-sensor bounds.")

    st.markdown("---")

    # 3. ANOMALY SCORE TREND CHART (Requirement 5)
    st.write(f"#### 3️⃣ Anomaly Score Trend Chart: {selected_m_id} ({curr_rec.get('Machine_Name', '')})")
    
    m_hist = st.session_state.history.get(selected_m_id, [])
    if m_hist:
        df_m_hist = pd.DataFrame(m_hist)
        df_m_hist["Step"] = range(len(df_m_hist))

        fig_anom = go.Figure()

        # Historical Anomaly Score Line
        fig_anom.add_trace(go.Scatter(
            x=df_m_hist["Step"],
            y=df_m_hist["Anomaly_Score"],
            mode="lines+markers",
            name="Anomaly Score (Isolation Forest)",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=4)
        ))

        # Highlight Anomaly Points in Red
        anom_points = df_m_hist[df_m_hist["Is_Anomaly"] == True]
        if not anom_points.empty:
            fig_anom.add_trace(go.Scatter(
                x=anom_points["Step"],
                y=anom_points["Anomaly_Score"],
                mode="markers",
                name="Flagged Anomaly Event",
                marker=dict(color="#DC2626", size=10, symbol="x")
            ))

        # 0.50 Decision Threshold Line
        fig_anom.add_hline(
            y=0.50,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text="Anomaly Decision Threshold (0.50)",
            annotation_position="top left"
        )

        fig_anom.update_layout(
            title=f"Continuous Anomaly Score Trajectory - {selected_m_id} (Contamination Cutoff: 0.05)",
            xaxis_title="Simulation Timeline (Steps / Days)",
            yaxis_title="Normalized Anomaly Score (0.0 = Normal, 1.0 = Extreme)",
            yaxis_range=[0.0, 1.05],
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Simulation history is building up. Click 'Start' or 'Next Step' to generate sensor telemetry.")


# ======================================================================================
# TAB 3: 🎯 XGBOOST RUL PREDICTION
# Supervised Remaining Useful Life Prediction (10 Sensors)
# ======================================================================================
with tabs[2]:
    st.subheader(f"🎯 XGBoost Remaining Useful Life (RUL) Prediction: {selected_m_id}")
    
    if xgb_model is None:
        st.error(f"⚠️ XGBoost RUL Model is unavailable: {xgb_error}. (Note: Anomaly detection continues to work independently!)")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="XGBoost Predicted RUL", value=f"{curr_rec.get('XGBoost_RUL', '--')} Days")
        with c2:
            st.metric(label="Ground-Truth RUL (Simulated)", value=f"{curr_rec.get('Remaining_Useful_Life_Days', '--')} Days")
        with c3:
            diff = "--"
            if curr_rec.get('XGBoost_RUL') is not None and curr_rec.get('Remaining_Useful_Life_Days') is not None:
                diff = f"{abs(curr_rec['XGBoost_RUL'] - curr_rec['Remaining_Useful_Life_Days'])} Days"
            st.metric(label="Prediction Error (|Actual - Pred|)", value=diff)

        # Plot RUL Degradation Trajectory
        m_hist = st.session_state.history.get(selected_m_id, [])
        if len(m_hist) > 1:
            df_rul = pd.DataFrame(m_hist)
            df_rul["Step"] = range(len(df_rul))

            fig_rul = go.Figure()
            fig_rul.add_trace(go.Scatter(
                x=df_rul["Step"],
                y=df_rul["XGBoost_RUL"],
                mode="lines",
                name="XGBoost Predicted RUL",
                line=dict(color="#2563EB", width=3)
            ))
            fig_rul.add_trace(go.Scatter(
                x=df_rul["Step"],
                y=df_rul["Remaining_Useful_Life_Days"],
                mode="lines",
                name="Ground Truth RUL",
                line=dict(color="#10B981", width=2, dash="dash")
            ))
            fig_rul.update_layout(
                title=f"RUL Trajectory (XGBoost Regressor) - {selected_m_id}",
                xaxis_title="Simulation Steps",
                yaxis_title="Remaining Useful Life (Days)",
                template="plotly_white",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rul, use_container_width=True)

        st.markdown("---")
        st.write("#### 🛠️ Industrial Servicing Guidance")
        rul_val = curr_rec.get('XGBoost_RUL', 200)
        if rul_val is not None:
            if rul_val > 150:
                st.success("🟢 **Action Guidance:** Machine is in nominal operating lifecycle. Routine condition monitoring recommended.")
            elif rul_val > 80:
                st.info("🔵 **Action Guidance:** Early wear detected. Schedule non-invasive acoustic and thermal inspection at next planned turnaround.")
            elif rul_val > 30:
                st.warning("🟠 **Action Guidance:** Progressive wear in bearing assembly. Order replacement spare parts and plan servicing within 2 weeks.")
            else:
                st.error("🔴 **Action Guidance:** CRITICAL - Terminal degradation stage. Immediate emergency maintenance required to avoid catastrophic shutdown.")


# ======================================================================================
# TAB 4: 📊 MULTI-SENSOR LIFECYCLE ANALYTICS
# ======================================================================================
with tabs[3]:
    st.subheader(f"📊 Multi-Sensor Telemetry Profiles: {selected_m_id}")
    
    m_hist = st.session_state.history.get(selected_m_id, [])
    if m_hist:
        df_hist = pd.DataFrame(m_hist)
        df_hist["Step"] = range(len(df_hist))

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            fig1 = px.line(df_hist, x="Step", y="Temperature", title="Temperature (°C)")
            fig1.update_traces(line_color="#EF4444")
            fig1.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = px.line(df_hist, x="Step", y="Motor_Current", title="Motor Current (A)")
            fig3.update_traces(line_color="#3B82F6")
            fig3.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig3, use_container_width=True)

            fig5 = px.line(df_hist, x="Step", y="Torque", title="Torque (Nm)")
            fig5.update_traces(line_color="#8B5CF6")
            fig5.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig5, use_container_width=True)

        with row1_col2:
            fig2 = px.line(df_hist, x="Step", y="Vibration", title="Vibration RMS (mm/s)")
            fig2.update_traces(line_color="#F59E0B")
            fig2.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig2, use_container_width=True)

            fig4 = px.line(df_hist, x="Step", y="Pressure", title="Pressure (bar)")
            fig4.update_traces(line_color="#10B981")
            fig4.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig4, use_container_width=True)

            fig6 = px.line(df_hist, x="Step", y="Acoustic_Emission", title="Acoustic Emission (dB)")
            fig6.update_traces(line_color="#EC4899")
            fig6.update_layout(template="plotly_white", height=230)
            st.plotly_chart(fig6, use_container_width=True)


# ======================================================================================
# TAB 5: 🔬 MACHINE LEARNING DIAGNOSTICS
# ======================================================================================
with tabs[4]:
    st.subheader("🔬 AI Engine Diagnostics & Benchmark Validation")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.write("#### 🎯 Supervised XGBoost RUL Regressor")
        results_path = os.path.join(base_dir, "models", "xgboost_experiment_results.json")
        if os.path.exists(results_path):
            import json
            with open(results_path, "r") as f:
                exp_data = json.load(f)
            
            xgb_metrics = exp_data.get("xgboost", {})
            st.write(f"- **Algorithm:** `{xgb_metrics.get('algorithm', 'XGBRegressor')}`")
            st.write(f"- **Trees / Estimators:** `{xgb_metrics.get('n_estimators', 500)}`")
            st.write(f"- **Learning Rate:** `{xgb_metrics.get('learning_rate', 0.05)}`")
            st.write(f"- **Test R² Score:** **{xgb_metrics.get('r2_score', 0.8915)}**")
            st.write(f"- **Mean Absolute Error (MAE):** **{xgb_metrics.get('mae', 25.73)} Days**")
            st.write(f"- **Root Mean Squared Error (RMSE):** **{xgb_metrics.get('rmse', 35.03)} Days**")

            # Feature Importance
            feat_imp = xgb_metrics.get("feature_importance", [])
            if feat_imp:
                imp_df = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)
                fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="XGBoost Feature Importance")
                fig_imp.update_layout(template="plotly_white", height=320)
                st.plotly_chart(fig_imp, use_container_width=True)

    with col_d2:
        st.write("#### 🛡️ Unsupervised Isolation Forest Anomaly Detector")
        st.write(f"- **Algorithm:** `scikit-learn IsolationForest`")
        st.write(f"- **Contamination Parameter:** `0.05` (5% expected anomaly proportion)")
        st.write(f"- **Number of Isolation Trees:** `150`")
        st.write(f"- **Feature Space:** `10 sensor parameters` (Excluding IDs and labels)")
        st.write(f"- **Model Hierarchy:** `Per-Machine Individual Models + Fleet Fallback`")
        st.write(f"- **Decision Score Formulation:** `s(x, n) = 2^(-E[h(x)] / c(n))`")

        st.info("""
        **Complementary Dual-Model Paradigm:**
        - **Isolation Forest:** Proactively catches acute, sudden, or uncharacteristic behavior (e.g. sensor drift, sudden cavitation, unexpected thermal spike) without requiring labels.
        - **XGBoost:** Projects long-term monotonic wear and estimates how many days of productive life remain before component replacement.
        """)
