import sys
import os
import asyncio
import logging
import uvicorn

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Ensure project modules are discoverable
base_dir = os.path.dirname(os.path.abspath(__file__))
for folder in ['simulator', 'preprocessing', 'model', 'utils']:
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.append(folder_path)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from simulator import FleetSimulator, RealTimeMachineSimulator
from predict import PredictiveMaintenanceModel, get_future_trend, filter_displayed_rul
from maintenance_engine import get_maintenance_recommendation
from preprocessing import load_data, preprocess_data
from utils import get_status_color

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_clock_loop())
    yield
    task.cancel()

app = FastAPI(title="Industrial Predictive Maintenance API - 8-Machine Fleet Engine", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = PredictiveMaintenanceModel()
sim_engine = FleetSimulator(max_lifespan_days=250, degradation_factor=1.8)

# Initialize unsupervised Isolation Forest Fleet Anomaly Detector
try:
    from isolation_forest import FleetAnomalyDetector
    anomaly_detector = FleetAnomalyDetector.load(os.path.join(base_dir, "models"))
except Exception as ex:
    logger.warning(f"Could not load FleetAnomalyDetector: {ex}")
    anomaly_detector = None

# Load CSV dataset exclusively for model evaluation metrics (/api/model)
df_eval = load_data()
df_eval.ffill(inplace=True)

class SimulationState:
    def __init__(self):
        self.auto_play = False
        self.simulation_speed = 1.0
        self.history_by_machine = {} # machine_id -> list of records
        self.fleet_ticks = [] # list of tick summaries
        self.prev_displayed_rul = {} # machine_id -> float
        self.prev_status_level = {} # machine_id -> int

    def reset(self):
        self.auto_play = False
        self.history_by_machine = {}
        self.fleet_ticks = []
        self.prev_displayed_rul = {}
        self.prev_status_level = {}

sim_state = SimulationState()

def reset_simulation_state():
    """Completely resets simulator engine, history buffers, predictions, and status state for all 8 machines."""
    sim_engine.reset()
    sim_state.reset()
    generate_next_telemetry_step()


def generate_next_telemetry_step():
    """
    Advances ALL 8 machines in one synchronized simulation tick, executes batch ML prediction,
    applies temporal smoothing per machine, and updates the Maintenance Decision Engine for the entire fleet.
    """
    fleet_raw_records = sim_engine.step()  # Returns 8 telemetry records
    
    # Extract feature matrix (8 rows x 10 features) for a single batch prediction call
    feature_batch = [
        {
            "Temperature": float(r.get("Temperature", 62.0)),
            "Vibration": float(r.get("Vibration", 0.20)),
            "Motor_Current": float(r.get("Motor_Current", 8.0)),
            "Pressure": float(r.get("Pressure", 5.0)),
            "RPM": float(r.get("RPM", 1750.0)),
            "Flow_Rate": float(r.get("Flow_Rate", 150.0)),
            "Oil_Temperature": float(r.get("Oil_Temperature", 45.0)),
            "Power_Consumption": float(r.get("Power_Consumption", 15.0)),
            "Torque": float(r.get("Torque", 81.86)),
            "Acoustic_Emission": float(r.get("Acoustic_Emission", 40.0)),
        }
        for r in fleet_raw_records
    ]
    
    # Single model batch prediction for all 8 machines
    raw_pred_ruls = model_service.predict_rul_batch(feature_batch)
    
    processed_fleet_records = []
    tick_num = sim_engine.tick
    
    for idx, record in enumerate(fleet_raw_records):
        m_id = record["Machine_ID"]
        temp = float(record.get("Temperature", 62.0))
        vib = float(record.get("Vibration", 0.20))
        curr = float(record.get("Motor_Current", 8.0))
        press = float(record.get("Pressure", 5.0))
        rpm_val = float(record.get("RPM", 1750.0))
        flow = float(record.get("Flow_Rate", 150.0))
        oil_temp = float(record.get("Oil_Temperature", 45.0))
        pwr = float(record.get("Power_Consumption", 15.0))
        torque = float(record.get("Torque", 81.86))
        acoustic_emission = float(record.get("Acoustic_Emission", 40.0))
        health = float(record.get("Machine_Health", 100.0))
        active_event = str(record.get("Active_Event", "None"))
        raw_pred_rul = raw_pred_ruls[idx]
        
        sim_inst = sim_engine.machines.get(m_id)
        temp_base = getattr(sim_inst, 'temp_base', 62.0) if sim_inst else 62.0
        vib_base = getattr(sim_inst, 'vib_base', 0.20) if sim_inst else 0.20
        curr_base = getattr(sim_inst, 'curr_base', 8.0) if sim_inst else 8.0
        press_base = getattr(sim_inst, 'press_base', 60.0) if sim_inst else 60.0
        rpm_base = getattr(sim_inst, 'rpm_base', 1750.0) if sim_inst else 1750.0
        flow_base = getattr(sim_inst, 'flow_base', 50.0) if sim_inst else 50.0
        oil_base = getattr(sim_inst, 'oil_base', 50.0) if sim_inst else 50.0
        pwr_base = getattr(sim_inst, 'power_base', 25.0) if sim_inst else 25.0
        max_lifespan = getattr(sim_inst, 'max_lifespan_days', 250) if sim_inst else 250
        
        t_dev = max(0.0, (temp - temp_base) / 15.0)
        v_dev = max(0.0, (vib - vib_base) / 4.0)
        c_dev = max(0.0, (curr - curr_base) / 7.2)
        p_dev = max(0.0, (press - press_base) / 20.0)
        r_dev = max(0.0, (rpm_base - rpm_val) / 300.0)
        f_dev = max(0.0, (flow_base - flow) / 20.0)
        o_dev = max(0.0, (oil_temp - oil_base) / 20.0)
        w_dev = max(0.0, (pwr - pwr_base) / 10.0)
        tq_dev = max(0.0, (torque - 81.86) / 20.0)
        ae_dev = max(0.0, (acoustic_emission - 40.0) / 15.0)
        sensor_anomaly_score = (0.16 * t_dev + 0.16 * v_dev + 0.12 * c_dev + 0.08 * p_dev + 0.08 * r_dev + 0.08 * f_dev + 0.08 * o_dev + 0.06 * w_dev + 0.09 * tq_dev + 0.09 * ae_dev)
        
        prev_rul = sim_state.prev_displayed_rul.get(m_id, None)
        val_float, displayed_rul = filter_displayed_rul(
            raw_prediction=raw_pred_rul,
            prev_displayed_rul=prev_rul,
            health=health
        )
        sim_state.prev_displayed_rul[m_id] = val_float
        
        prev_lvl = sim_state.prev_status_level.get(m_id, 0)
        maint_info = get_maintenance_recommendation(
            predicted_rul=displayed_rul,
            machine_health=health,
            active_event=active_event,
            max_lifespan_days=max_lifespan,
            prev_status_level=prev_lvl,
            sensor_anomaly_score=sensor_anomaly_score,
            temperature=temp,
            vibration=vib,
            motor_current=curr,
            pressure=press,
            rpm=rpm_val,
            flow_rate=flow,
            oil_temperature=oil_temp,
            power_consumption=pwr,
            torque=torque,
            acoustic_emission=acoustic_emission,
            temp_base=temp_base,
            vib_base=vib_base,
            curr_base=curr_base,
            press_base=press_base,
            rpm_base=rpm_base,
            flow_base=flow_base,
            oil_temp_base=oil_base,
            power_base=pwr_base
        )
        sim_state.prev_status_level[m_id] = maint_info.get("status_level", 0)
        
        record["Raw_Predicted_RUL"] = raw_pred_rul
        record["Predicted_RUL"] = displayed_rul
        record["Machine_Status"] = maint_info["maintenance_status"]
        record["Recommended_Action"] = maint_info["recommended_action"]
        record["Inspection_Priority"] = maint_info["inspection_priority"]
        record["Next_Inspection_Window"] = maint_info["next_inspection_window"]
        record["Abnormal_Indicators"] = maint_info.get("abnormal_indicators", [])

        # Unsupervised Isolation Forest Anomaly Detection
        if anomaly_detector:
            try:
                anom_eval = anomaly_detector.predict_reading(record, machine_id=m_id)
                record["Anomaly_Label"] = anom_eval.get("anomaly_label", "normal")
                record["Anomaly_Score"] = anom_eval.get("anomaly_score", 0.0)
                record["Anomaly_Severity"] = anom_eval.get("anomaly_severity", "Normal")
                record["Is_Anomaly"] = anom_eval.get("is_anomaly", False)
                record["Anomaly_Model"] = anom_eval.get("model_type", "per_machine")
            except Exception:
                record["Anomaly_Label"] = "normal"
                record["Anomaly_Score"] = 0.0
                record["Anomaly_Severity"] = "Normal"
                record["Is_Anomaly"] = False
                record["Anomaly_Model"] = "none"
        
        if m_id not in sim_state.history_by_machine:
            sim_state.history_by_machine[m_id] = []
        sim_state.history_by_machine[m_id].append(record)
        if len(sim_state.history_by_machine[m_id]) > 1000:
            sim_state.history_by_machine[m_id].pop(0)
            
        processed_fleet_records.append(record)
        
    sim_state.fleet_ticks.append({
        "tick": tick_num,
        "timestamp": processed_fleet_records[0]["Timestamp"],
        "machines": processed_fleet_records
    })
    if len(sim_state.fleet_ticks) > 1000:
        sim_state.fleet_ticks.pop(0)
        
    return processed_fleet_records


# Generate initial baseline step on server startup
if not sim_state.fleet_ticks:
    generate_next_telemetry_step()

async def simulation_clock_loop():
    """
    Backend simulation clock loop.
    Continuously ticks the 8-machine fleet simulator according to sim_state.simulation_speed
    when sim_state.auto_play is True.
    """
    while True:
        try:
            if sim_state.auto_play:
                generate_next_telemetry_step()
                sleep_time = max(0.05, float(sim_state.simulation_speed))
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"Error in simulation_clock_loop: {e}", exc_info=True)
            await asyncio.sleep(0.5)

@app.get("/api/status")
def get_status():
    try:
        return {
            "status": "online",
            "fleet_size": len(sim_engine.machines),
            "machines": [cfg["machine_id"] for cfg in FleetSimulator.MACHINE_CONFIGS],
            "data_source": "8-Machine Synchronized Fleet SCADA Simulator",
            "model": "Random Forest Regressor (Batch Inference)",
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error in GET /api/status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch system status: {str(e)}")

@app.get("/api/fleet")
def get_fleet():
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
        latest_tick = sim_state.fleet_ticks[-1]
        return {
            "tick": latest_tick["tick"],
            "timestamp": latest_tick["timestamp"],
            "fleet_size": len(latest_tick["machines"]),
            "machines": latest_tick["machines"]
        }
    except Exception as e:
        logger.error(f"Error in GET /api/fleet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch fleet data: {str(e)}")

@app.get("/api/current")
def get_current(machine_id: str = None):
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
            
        current_tick = sim_state.fleet_ticks[-1]
        machines_list = current_tick["machines"]
        
        selected_machine = None
        if machine_id:
            selected_machine = next((m for m in machines_list if m["Machine_ID"] == machine_id), None)
        if not selected_machine:
            selected_machine = machines_list[0]
            
        fleet_size = len(machines_list)
        healthy_cnt = sum(1 for m in machines_list if m["Machine_Status"] == "Healthy")
        monitor_cnt = sum(1 for m in machines_list if m["Machine_Status"] in ["Slight Wear", "Moderate Wear"])
        warning_cnt = sum(1 for m in machines_list if m["Machine_Status"] == "Warning")
        critical_cnt = sum(1 for m in machines_list if m["Machine_Status"] == "Critical")

        high_prio_cnt = sum(1 for m in machines_list if m["Inspection_Priority"] == "High")
        urgent_prio_cnt = sum(1 for m in machines_list if m["Inspection_Priority"] == "Urgent")
        critical_prio_cnt = sum(1 for m in machines_list if m["Inspection_Priority"] == "Critical")

        avg_health = round(float(np.mean([m["Machine_Health"] for m in machines_list])), 1)
        avg_rul = int(round(float(np.mean([m["Predicted_RUL"] for m in machines_list]))))
        critical_alerts = urgent_prio_cnt + critical_prio_cnt

        fleet_kpi = {
            "total_machines": fleet_size,
            "healthy_count": healthy_cnt,
            "monitor_count": monitor_cnt,
            "warning_count": warning_cnt,
            "critical_count": critical_cnt,
            "high_priority_count": high_prio_cnt,
            "urgent_priority_count": urgent_prio_cnt,
            "critical_priority_count": critical_prio_cnt,
            "avg_health": avg_health,
            "avg_predicted_rul": avg_rul,
            "critical_alerts_count": critical_alerts
        }

        return {
            "tick": current_tick["tick"],
            "current_idx": len(sim_state.fleet_ticks) - 1,
            "total_records": max(1000, len(sim_state.fleet_ticks)),
            "timestamp": str(current_tick["timestamp"]),
            "fleet_size": fleet_size,
            "fleet_avg_health": avg_health,
            "fleet_kpi": fleet_kpi,
            "auto_play": sim_state.auto_play,
            "simulation_speed": sim_state.simulation_speed,
            "machines": machines_list,
            # Selected machine focus detail fields (backward compatible)
            "machine_id": selected_machine["Machine_ID"],
            "machine_name": selected_machine.get("Machine_Name", selected_machine["Machine_ID"]),
            "temperature": round(float(selected_machine.get("Temperature", 62.0)), 2),
            "vibration": round(float(selected_machine.get("Vibration", 0.20)), 2),
            "motor_current": round(float(selected_machine.get("Motor_Current", 8.0)), 2),
            "pressure": round(float(selected_machine.get("Pressure", 5.0)), 2),
            "rpm": round(float(selected_machine.get("RPM", 1750.0)), 2),
            "flow_rate": round(float(selected_machine.get("Flow_Rate", 150.0)), 2),
            "oil_temperature": round(float(selected_machine.get("Oil_Temperature", 45.0)), 2),
            "power_consumption": round(float(selected_machine.get("Power_Consumption", 15.0)), 2),
            "torque": round(float(selected_machine.get("Torque", 81.86)), 2),
            "acoustic_emission": round(float(selected_machine.get("Acoustic_Emission", 40.0)), 2),
            "machine_health": round(float(selected_machine["Machine_Health"]), 1),
            "machine_status": str(selected_machine["Machine_Status"]),
            "actual_rul_days": int(selected_machine["Remaining_Useful_Life_Days"]),
            "predicted_rul_days": int(selected_machine["Predicted_RUL"]),
            "alert_status": str(selected_machine["Machine_Status"]),
            "active_event": str(selected_machine.get("Active_Event", "None")),
            "abnormal_indicators": selected_machine.get("Abnormal_Indicators", [])
        }
    except Exception as e:
        logger.error(f"Error in GET /api/current: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch current telemetry: {str(e)}")

@app.get("/api/machines")
def get_machines():
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
        return {"machines": sim_state.fleet_ticks[-1]["machines"]}
    except Exception as e:
        logger.error(f"Error in GET /api/machines: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch machines: {str(e)}")

@app.get("/api/machines/{machine_id}")
def get_machine_by_id(machine_id: str):
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
        m_record = next((m for m in sim_state.fleet_ticks[-1]["machines"] if m["Machine_ID"] == machine_id), None)
        if not m_record:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
        return m_record
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GET /api/machines/{machine_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch machine detail: {str(e)}")

@app.get("/api/history")
def get_history(machine_id: str = "M-001"):
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
            
        m_history = sim_state.history_by_machine.get(machine_id)
        if not m_history:
            m_history = sim_state.history_by_machine.get("M-001", [])
            
        records = m_history.copy()
        
        def make_trend_dict(col_name):
            vals = [float(r.get(col_name, 0.0)) for r in records]
            if not vals:
                vals = [0.0]
            history_window = vals[-50:] if len(vals) > 50 else vals
            future_vals = get_future_trend(history_window, steps=20)
            return {
                "actual": vals,
                "predicted_future": future_vals
            }
            
        fleet_comparison = {}
        for m_id, hist in sim_state.history_by_machine.items():
            fleet_comparison[m_id] = {
                "temperature": [r.get("Temperature", 0.0) for r in hist],
                "vibration": [r.get("Vibration", 0.0) for r in hist],
                "motor_current": [r.get("Motor_Current", 0.0) for r in hist],
                "pressure": [r.get("Pressure", 0.0) for r in hist],
                "rpm": [r.get("RPM", 0.0) for r in hist],
                "flow_rate": [r.get("Flow_Rate", 0.0) for r in hist],
                "oil_temperature": [r.get("Oil_Temperature", 0.0) for r in hist],
                "power_consumption": [r.get("Power_Consumption", 0.0) for r in hist],
                "torque": [r.get("Torque", 0.0) for r in hist],
                "acoustic_emission": [r.get("Acoustic_Emission", 0.0) for r in hist],
                "machine_health": [r.get("Machine_Health", 0.0) for r in hist],
                "predicted_rul": [r.get("Predicted_RUL", 0) for r in hist]
            }
            
        return {
            "machine_id": machine_id,
            "records": records,
            "temperature_trend": make_trend_dict("Temperature"),
            "vibration_trend": make_trend_dict("Vibration"),
            "motor_current_trend": make_trend_dict("Motor_Current"),
            "pressure_trend": make_trend_dict("Pressure"),
            "rpm_trend": make_trend_dict("RPM"),
            "flow_rate_trend": make_trend_dict("Flow_Rate"),
            "oil_temperature_trend": make_trend_dict("Oil_Temperature"),
            "power_consumption_trend": make_trend_dict("Power_Consumption"),
            "torque_trend": make_trend_dict("Torque"),
            "acoustic_emission_trend": make_trend_dict("Acoustic_Emission"),
            "machine_health_trend": make_trend_dict("Machine_Health"),
            "rul_trend": make_trend_dict("Predicted_RUL"),
            "fleet_comparison": fleet_comparison
        }
    except Exception as e:
        logger.error(f"Error in GET /api/history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history trends: {str(e)}")

@app.get("/api/maintenance")
def get_maintenance_status():
    try:
        if not sim_state.fleet_ticks:
            generate_next_telemetry_step()
            
        current_machines = sim_state.fleet_ticks[-1]["machines"]
        maintenance_records = []
        for m in current_machines:
            maintenance_records.append({
                "machine_id": m["Machine_ID"],
                "machine_name": m.get("Machine_Name", m["Machine_ID"]),
                "predicted_rul_days": m["Predicted_RUL"],
                "machine_health": m["Machine_Health"],
                "maintenance_status": m["Machine_Status"],
                "recommended_action": m["Recommended_Action"],
                "inspection_priority": m["Inspection_Priority"],
                "next_inspection_window": m["Next_Inspection_Window"],
                "abnormal_indicators": m.get("Abnormal_Indicators", []),
                "timestamp": m["Timestamp"],
                "temperature": m.get("Temperature", 0.0),
                "vibration": m.get("Vibration", 0.0),
                "motor_current": m.get("Motor_Current", 0.0),
                "pressure": m.get("Pressure", 0.0),
                "rpm": m.get("RPM", 0.0),
                "flow_rate": m.get("Flow_Rate", 0.0),
                "oil_temperature": m.get("Oil_Temperature", 0.0),
                "power_consumption": m.get("Power_Consumption", 0.0),
                "torque": m.get("Torque", 0.0),
                "acoustic_emission": m.get("Acoustic_Emission", 0.0)
            })
            
        prio_order = {"Critical": 0, "Urgent": 1, "High": 2, "Moderate": 3, "Low": 4}
        maintenance_records.sort(key=lambda x: prio_order.get(x["inspection_priority"], 5))
        
        top_critical = maintenance_records[0] if maintenance_records else {}
        
        return {
            "fleet_size": len(maintenance_records),
            "maintenance_list": maintenance_records,
            # Single object backward compatibility
            "machine_id": top_critical.get("machine_id", "M-001"),
            "predicted_rul_days": top_critical.get("predicted_rul_days", 0),
            "machine_health": top_critical.get("machine_health", 100.0),
            "maintenance_status": top_critical.get("maintenance_status", "Healthy"),
            "recommended_action": top_critical.get("recommended_action", "Continue Normal Operation"),
            "inspection_priority": top_critical.get("inspection_priority", "Low"),
            "next_inspection_window": top_critical.get("next_inspection_window", "Routine inspection within 90-120 days"),
            "timestamp": top_critical.get("timestamp", ""),
            "temperature": top_critical.get("temperature", 0.0),
            "vibration": top_critical.get("vibration", 0.0),
            "motor_current": top_critical.get("motor_current", 0.0),
            "pressure": top_critical.get("pressure", 0.0),
            "rpm": top_critical.get("rpm", 0.0),
            "flow_rate": top_critical.get("flow_rate", 0.0),
            "oil_temperature": top_critical.get("oil_temperature", 0.0),
            "power_consumption": top_critical.get("power_consumption", 0.0),
            "torque": top_critical.get("torque", 0.0),
            "acoustic_emission": top_critical.get("acoustic_emission", 0.0)
        }
    except Exception as e:
        logger.error(f"Error in GET /api/maintenance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch maintenance status: {str(e)}")


class PredictRequest(BaseModel):
    temperature: float = 62.0
    vibration: float = 0.20
    motor_current: float = 8.0
    pressure: float = 5.0
    rpm: float = 1750.0
    flow_rate: float = 150.0
    oil_temperature: float = 45.0
    power_consumption: float = 15.0
    torque: float = 81.86
    acoustic_emission: float = 40.0

@app.post("/api/predict")
def predict_rul(req: PredictRequest):
    try:
        rul = model_service.predict_rul(
            req.temperature, req.vibration, req.motor_current,
            req.pressure, req.rpm, req.flow_rate, req.oil_temperature, req.power_consumption,
            req.torque, req.acoustic_emission
        )
        return {"predicted_rul": rul}
    except Exception as e:
        logger.error(f"Error in POST /api/predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to calculate RUL prediction: {str(e)}")

@app.get("/api/model")
def get_model_evaluation():
    try:
        if not model_service.model or not model_service.scaler:
            raise HTTPException(status_code=500, detail="ML Model not loaded")
            
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df_eval, is_training=True)
        y_pred = model_service.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        feature_names = [
            "Temperature", "Vibration", "Motor_Current",
            "Pressure", "RPM", "Flow_Rate", "Oil_Temperature", "Power_Consumption",
            "Torque", "Acoustic_Emission"
        ]
        importances = model_service.model.feature_importances_.tolist()
        feature_importance = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(feature_names, importances)
        ]
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
        actual_samples = [round(float(x), 1) for x in y_test.iloc[sample_indices].tolist()]
        predicted_samples = [round(float(x), 1) for x in y_pred[sample_indices].tolist()]
        residuals = [round(float(a - p), 2) for a, p in zip(actual_samples, predicted_samples)]
        
        # Load precomputed XGBoost experiment results if available (NO ONLINE TRAINING)
        xgb_exp_data = None
        xgb_json_path = os.path.join(base_dir, "models", "xgboost_experiment_results.json")
        if os.path.exists(xgb_json_path):
            try:
                with open(xgb_json_path, "r") as f:
                    xgb_exp_data = json.load(f)
            except Exception as ex:
                logger.warning(f"Could not load xgboost_experiment_results.json: {ex}")

        return {
            "production_model": "Random Forest Regressor (10-Feature)",
            "algorithm": "Random Forest Regressor",
            "n_estimators": 100,
            "dataset_size": len(df_eval),
            "train_size": len(X_train_scaled),
            "test_size": len(X_test_scaled),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2_score": round(float(r2), 4),
            "feature_importance": feature_importance,
            "scatter_plot": {
                "actual": actual_samples,
                "predicted": predicted_samples
            },
            "residuals": residuals,
            "xgboost_experiment": xgb_exp_data
        }
    except Exception as e:
        logger.error(f"Error in GET /api/model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch model metrics: {str(e)}")


@app.get("/api/logs")
def get_logs(machine_id: str = "M-001"):
    try:
        m_history = sim_state.history_by_machine.get(machine_id, [])
        recent = m_history[-10:] if m_history else []
        logs = []
        for r in recent:
            temp = float(r.get("Temperature", 62.0))
            vib = float(r.get("Vibration", 0.20))
            curr = float(r.get("Motor_Current", 8.0))
            press = float(r.get("Pressure", 5.0))
            rpm = float(r.get("RPM", 1750.0))
            flow = float(r.get("Flow_Rate", 150.0))
            oil_temp = float(r.get("Oil_Temperature", 45.0))
            power = float(r.get("Power_Consumption", 15.0))
            torque = float(r.get("Torque", 81.86))
            acoustic = float(r.get("Acoustic_Emission", 40.0))
            health = float(r.get("Machine_Health", 100.0))
            pred_rul = int(r.get("Predicted_RUL", 0))
            status, _ = get_status_color(health)
            
            logs.append({
                "timestamp": str(r.get("Timestamp", "")),
                "temperature": round(temp, 2),
                "vibration": round(vib, 2),
                "motor_current": round(curr, 2),
                "pressure": round(press, 2),
                "rpm": round(rpm, 2),
                "flow_rate": round(flow, 2),
                "oil_temperature": round(oil_temp, 2),
                "power_consumption": round(power, 2),
                "torque": round(torque, 2),
                "acoustic_emission": round(acoustic, 2),
                "predicted_rul": pred_rul,
                "machine_health": round(health, 1),
                "status": status,
                "active_event": r.get("Active_Event", "None")
            })
            
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error in GET /api/logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent logs: {str(e)}")

class ControlAction(BaseModel):
    action: str  # "start", "pause", "next", "reset", "set_speed"
    speed: float = 1.0

@app.post("/api/control")
def control_simulation(action: ControlAction):
    try:
        if action.action == "start":
            sim_state.auto_play = True
        elif action.action == "pause":
            sim_state.auto_play = False
        elif action.action == "next":
            generate_next_telemetry_step()
        elif action.action == "reset":
            reset_simulation_state()
        elif action.action == "set_speed":
            sim_state.simulation_speed = action.speed
                
        return {
            "current_idx": len(sim_state.fleet_ticks) - 1,
            "auto_play": sim_state.auto_play,
            "simulation_speed": sim_state.simulation_speed
        }

    except Exception as e:
        logger.error(f"Error in POST /api/control: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process control action: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

