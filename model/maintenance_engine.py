STATUS_HIERARCHY = ["Healthy", "Slight Wear", "Moderate Wear", "Warning", "Critical"]

def get_maintenance_recommendation(
    predicted_rul,
    machine_health=100.0,
    active_event="None",
    max_lifespan_days=250,
    prev_status_level=0,
    sensor_anomaly_score=0.0,
    temperature=None,
    vibration=None,
    motor_current=None,
    pressure=None,
    rpm=None,
    flow_rate=None,
    oil_temperature=None,
    power_consumption=None,
    torque=None,
    acoustic_emission=None,
    temp_base=62.0,
    vib_base=0.20,
    curr_base=8.0,
    press_base=5.0,
    rpm_base=1750.0,
    flow_base=150.0,
    oil_temp_base=45.0,
    power_base=15.0,
    torque_base=81.86,
    acoustic_base=40.0,
    **kwargs
):
    """
    Multi-Factor Maintenance Decision Engine (Single Source of Truth).
    Evaluates:
      1. Machine Health Index (%)
      2. Displayed Remaining Useful Life (RUL in days)
      3. Sensor anomaly severity score
      4. Persistent active event status
      5. Real sensor telemetry deviations across all 8 canonical features
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

    # Real telemetry deviation indicators based on physical machine baselines across all 8 features
    abnormal_indicators = []
    if temperature is not None:
        t_val = float(temperature)
        t_b = float(temp_base)
        t_diff = t_val - t_b
        if t_diff >= 6.0 or t_val >= 75.0:
            abnormal_indicators.append(f"Elevated Temperature: {t_val:.1f}°C (+{t_diff:.1f}°C vs baseline)")
            
    if vibration is not None:
        v_val = float(vibration)
        v_b = float(vib_base)
        v_diff = v_val - v_b
        if v_diff >= 0.40 or v_val >= 1.0:
            abnormal_indicators.append(f"High Vibration: {v_val:.2f} mm/s (+{v_diff:.2f} mm/s vs baseline)")

    if motor_current is not None:
        c_val = float(motor_current)
        c_b = float(curr_base)
        c_diff = c_val - c_b
        if c_diff >= 2.0 or c_val >= 11.5:
            abnormal_indicators.append(f"Elevated Motor Current: {c_val:.1f} A (+{c_diff:.1f} A vs baseline)")

    if pressure is not None:
        p_val = float(pressure)
        p_b = float(press_base)
        p_diff = p_val - p_b
        if abs(p_diff) >= 1.0 or p_val < 3.8 or p_val > 6.5:
            diff_str = f"+{p_diff:.2f}" if p_diff >= 0 else f"{p_diff:.2f}"
            abnormal_indicators.append(f"Operating Pressure Deviation: {p_val:.2f} bar ({diff_str} bar vs baseline)")

    if rpm is not None:
        r_val = float(rpm)
        r_b = float(rpm_base)
        r_diff = r_val - r_b
        if abs(r_diff) >= 60.0 or r_val < 1680.0:
            diff_str = f"+{r_diff:.0f}" if r_diff >= 0 else f"{r_diff:.0f}"
            abnormal_indicators.append(f"Shaft Speed Deviation: {r_val:.0f} RPM ({diff_str} RPM vs baseline)")

    if flow_rate is not None:
        f_val = float(flow_rate)
        f_b = float(flow_base)
        f_diff = f_val - f_b
        if f_diff <= -15.0 or f_val < 130.0:
            abnormal_indicators.append(f"Flow Rate Below Normal: {f_val:.1f} L/min ({f_diff:.1f} L/min vs baseline)")

    if oil_temperature is not None:
        ot_val = float(oil_temperature)
        ot_b = float(oil_temp_base)
        ot_diff = ot_val - ot_b
        if ot_diff >= 10.0 or ot_val >= 58.0:
            abnormal_indicators.append(f"Oil Temperature Elevated: {ot_val:.1f}°C (+{ot_diff:.1f}°C vs baseline)")

    if power_consumption is not None:
        pw_val = float(power_consumption)
        pw_b = float(power_base)
        pw_diff = pw_val - pw_b
        if pw_diff >= 5.0 or pw_val >= 22.0:
            abnormal_indicators.append(f"Power Demand Elevated: {pw_val:.1f} kW (+{pw_diff:.1f} kW vs baseline)")

    if torque is not None:
        tq_val = float(torque)
        tq_b = float(torque_base)
        tq_diff = tq_val - tq_b
        if tq_diff >= 15.0 or tq_val >= 105.0:
            abnormal_indicators.append(f"High Shaft Torque: {tq_val:.1f} Nm (+{tq_diff:.1f} Nm vs baseline)")

    if acoustic_emission is not None:
        ae_val = float(acoustic_emission)
        ae_b = float(acoustic_base)
        ae_diff = ae_val - ae_b
        if ae_diff >= 8.0 or ae_val >= 52.0:
            abnormal_indicators.append(f"High Acoustic Emission: {ae_val:.1f} dB (+{ae_diff:.1f} dB vs baseline)")

    if active_event not in ["None", "", None]:
        abnormal_indicators.append(f"Fault Signal: {active_event}")

    return {
        "maintenance_status": status,
        "recommended_action": recommendation,
        "inspection_priority": priority,
        "next_inspection_window": window,
        "status_level": current_level,
        "abnormal_indicators": abnormal_indicators
    }


