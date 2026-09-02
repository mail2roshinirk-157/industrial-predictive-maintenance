STATUS_HIERARCHY = ["Healthy", "Slight Wear", "Moderate Wear", "Warning", "Critical"]

def get_maintenance_recommendation(
    predicted_rul,
    machine_health=100.0,
    active_event="None",
    max_lifespan_days=250,
    prev_status_level=0,
    sensor_anomaly_score=0.0
):
    """
    Multi-Factor Maintenance Decision Engine (Single Source of Truth).
    Evaluates:
      1. Machine Health Index (%)
      2. Displayed Remaining Useful Life (RUL in days)
      3. Sensor anomaly severity score
      4. Persistent active event status
    Enforces non-regressive monotonic status transitions (status level never decreases during a run).
    Prevents premature Critical status (e.g. Health 38% & RUL 76 is Warning/Moderate Wear, never Critical).
    """
    rul = float(predicted_rul)
    health = float(machine_health)
    lifespan = max(1.0, float(max_lifespan_days))
    anomaly = float(sensor_anomaly_score)
    
    # Proportional RUL thresholds relative to lifespan
    rul_crit = 0.10 * lifespan   # ~25 days for 250-day lifespan
    rul_warn = 0.25 * lifespan   # ~62 days for 250-day lifespan
    rul_mod = 0.50 * lifespan    # ~125 days for 250-day lifespan
    rul_slight = 0.75 * lifespan # ~187 days for 250-day lifespan
    
    is_severe_fault = active_event not in ["None", "", None] and ("Burst" in active_event or "Peak" in active_event)

    # Multi-Factor Stage Evaluation Policy
    if health <= 15.0 or rul <= rul_crit or (health <= 25.0 and is_severe_fault):
        calculated_level = 4  # Critical (Imminent failure or severe end-of-life)
    elif health < 40.0 or rul < rul_warn or anomaly > 1.5 or is_severe_fault:
        calculated_level = 3  # Warning
    elif health < 60.0 or rul < rul_mod or anomaly > 0.8:
        calculated_level = 2  # Moderate Wear
    elif health < 80.0 or rul <= rul_slight or anomaly > 0.3:
        calculated_level = 1  # Slight Wear
    else:
        calculated_level = 0  # Healthy

    # Enforce monotonic status level (status level must never regress unless Reset is executed)
    current_level = max(prev_status_level, calculated_level)
    status = STATUS_HIERARCHY[current_level]
    
    if status == "Critical":
        recommendation = "Immediate maintenance required"
        priority = "Critical"
        window = "Stop machine immediately before catastrophic failure"
    elif status == "Warning":
        recommendation = "Schedule maintenance urgently"
        priority = "Urgent"
        window = "Perform servicing within 3–5 days"
    elif status == "Moderate Wear":
        recommendation = "Schedule preventive maintenance"
        priority = "High"
        window = "Schedule maintenance within 14–21 days"
    elif status == "Slight Wear":
        recommendation = "Increase monitoring frequency"
        priority = "Moderate"
        window = "Inspect within 30–45 days"
    else:
        status = "Healthy"
        recommendation = "Continue Normal Operation"
        priority = "Low"
        window = "Routine inspection within 90–120 days"

    # Override priority for active anomaly events without falsely forcing Critical status
    if active_event not in ["None", "", None] and priority in ["Low", "Moderate"]:
        priority = "High"
        recommendation = f"Active anomaly ({active_event}): Inspect component immediately"
        window = "Inspect within 24-48 hours"


    return {
        "maintenance_status": status,
        "recommended_action": recommendation,
        "inspection_priority": priority,
        "next_inspection_window": window,
        "status_level": current_level
    }

