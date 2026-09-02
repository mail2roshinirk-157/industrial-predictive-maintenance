import numpy as np
import datetime
import random

class RealTimeMachineSimulator:
    """
    Industrial Real-Time Machine Telemetry Simulator (365-Day Digital Twin SCADA Model).
    Models realistic industrial condition monitoring telemetry across 4 operational phases:
      - Phase 1 (Days 1–150, Healthy Operation): Stationary stochastic process with random walk micro-drift + Gaussian noise (zero upward trend). No flat lines.
      - Phase 2 (Days 151–260, Early Wear): Non-monotonic mean rise + variance growth.
      - Phase 3 (Days 261–330, Progressive Degradation): Accelerated wear, vibration oscillations, overload current peaks.
      - Phase 4 (Days 331–365+, Critical Stage): Severe instability, highest signal noise, random thermal spikes, health collapse.

    Digital Twin Health Concept:
      Health Index emerges dynamically from a cumulative multi-sensor wear score combining age,
      temperature deviation, vibration severity, motor current, acoustic noise, and signal instability.
      Strictly clamped between 0% and 100%.
    """
    def __init__(self, max_lifespan_days=250, degradation_factor=1.8, seed=None, degradation_start_day=None, degradation_speed=1.0):
        self.max_lifespan_days = max_lifespan_days
        self.degradation_factor = degradation_factor
        self.seed = seed
        self.degradation_start_day = degradation_start_day
        self.degradation_speed = degradation_speed
        self.reset(seed=self.seed)

    def reset(self, seed=None):
        """Resets the machine back to initial baseline state with unique machine parameters."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        self.health = 100.0
        self.prev_health = 100.0
        self.cum_wear = 0.0  # Monotonic physical wear tracking accumulator
        self.start_time = datetime.datetime.now()
        
        # 1. Target lifespan scaled proportionally to max_lifespan_days (~230 to 270 days for 250-day baseline)
        min_span = int(self.max_lifespan_days * 0.92)
        max_span = int(self.max_lifespan_days * 1.08)
        self.target_lifespan = random.randint(min_span, max_span)
        self.rul = float(self.target_lifespan)
        
        # 2. Wear Onset Day (Proportional to target_lifespan, ~0.40 * target_lifespan)
        if self.degradation_start_day is not None:
            self.start_wear_day = float(self.degradation_start_day) * (float(self.target_lifespan) / float(self.max_lifespan_days))
        else:
            self.start_wear_day = round(float(self.target_lifespan) * random.uniform(0.38, 0.42), 1)
            
        # 3. Degradation Speed & Variance Coefficients
        self.speed = float(self.degradation_speed) * random.uniform(0.85, 1.15)
        self.var_coeff = random.uniform(0.85, 1.15)
        
        # 4. Statistically unique baseline operating parameters
        self.temp_base = random.uniform(61.2, 62.8)   # °C
        self.vib_base = random.uniform(0.18, 0.22)    # mm/s
        self.curr_base = random.uniform(7.8, 8.2)     # A
        self.noise_base = random.uniform(41.0, 43.0)  # dB
        
        # Random walk micro-drift initial state (Brownian motion for SCADA realism)
        self.temp_drift = 0.0
        self.vib_drift = 0.0
        self.curr_drift = 0.0
        self.noise_drift = 0.0
        
        # Base noise standard deviations during Healthy phase
        self.sigma_temp_base = random.uniform(0.22, 0.32)
        self.sigma_vib_base = random.uniform(0.018, 0.025)
        self.sigma_curr_base = random.uniform(0.035, 0.055)
        self.sigma_noise_base = random.uniform(0.45, 0.65)
        
        # Statistically unique degradation sensitivity rates
        self.k_temp = random.uniform(0.85, 1.15) * self.speed
        self.k_vib = random.uniform(0.85, 1.15) * self.speed
        self.k_curr = random.uniform(0.85, 1.15) * self.speed
        self.k_noise = random.uniform(0.85, 1.15) * self.speed
        
        self.spike_prob = random.uniform(0.15, 0.25)
        self.spike_mag = random.uniform(0.85, 1.25)
        
        self.active_event = "None"

    def _evaluate_stage(self):
        """Determines machine operational stage based on current health percentage."""
        if self.health >= 80.0:
            return "Healthy"
        elif self.health >= 60.0:
            return "Slight Wear"
        elif self.health >= 40.0:
            return "Moderate Wear"
        elif self.health >= 15.0:
            return "Critical"
        else:
            return "Failure"

    def step(self):
        """
        Advances the machine simulation by 1 step/day.
        Returns telemetry dictionary with stochastic living data, correlated multi-sensor signals,
        and monotonic cumulative health decay.
        """
        self.current_step += 1
        day = float(self.current_step)
        
        # 1. Low-frequency stochastic drift update (zero flat lines)
        self.temp_drift = 0.82 * self.temp_drift + np.random.normal(0, 0.15)
        self.vib_drift = 0.82 * self.vib_drift + np.random.normal(0, 0.010)
        self.curr_drift = 0.82 * self.curr_drift + np.random.normal(0, 0.025)
        self.noise_drift = 0.82 * self.noise_drift + np.random.normal(0, 0.30)
        
        # 2. Smooth Continuous Phase Blending (Sigmoid transition around start_wear_day)
        # Eliminates step-discontinuities between Healthy and Wear phases
        wear_onset = self.start_wear_day
        k_blend = 0.08
        wear_blend = 1.0 / (1.0 + np.exp(-k_blend * (day - wear_onset)))
        
        # Normalized wear progression (0.0 during healthy, increasing up to ~1.3+ past lifespan)
        wear_span = max(1.0, float(self.target_lifespan) - wear_onset)
        p = max(0.0, (day - wear_onset) / wear_span) * wear_blend
        
        # Micro-oscillations & diurnal cycles
        t_osc = 0.22 * np.sin(day * 0.12 * np.pi) + 0.14 * np.cos(day * 0.04 * np.pi)
        v_osc = 0.014 * np.sin(day * 0.15 * np.pi)
        
        # Mean trend calculation with smooth continuous transition
        temp_mean = self.temp_base + t_osc + self.temp_drift * 0.3 + (16.5 * self.k_temp) * (p ** 1.7) + 0.40 * np.sin(p * 8 * np.pi)
        vib_mean = self.vib_base + v_osc + self.vib_drift * 0.3 + (4.3 * self.k_vib) * (p ** 1.7) + 0.16 * np.sin(p * 10 * np.pi)
        curr_mean = self.curr_base + self.curr_drift * 0.3 + (8.2 * self.k_curr) * (p ** 1.7)
        noise_mean = self.noise_base + self.noise_drift * 0.3 + (33.0 * self.k_noise) * (p ** 1.7)
        
        # Heteroskedastic noise scaling with wear (variance grows with age)
        sigma_t = (self.sigma_temp_base + 2.3 * (p ** 1.5)) * self.var_coeff
        sigma_v = (self.sigma_vib_base + 0.38 * (p ** 1.8)) * self.var_coeff
        sigma_c = (self.sigma_curr_base + 0.48 * (p ** 1.5)) * self.var_coeff
        sigma_n = (self.sigma_noise_base + 4.2 * (p ** 1.5)) * self.var_coeff
        
        # Coupled multi-sensor overload events & random temporary recovery dips
        spike_factor = 0.0
        if p > 0.20 and random.random() < self.spike_prob:
            spike_factor = random.uniform(0.8, 1.3) * self.spike_mag
            self.active_event = "Thermal Burst" if random.random() < 0.5 else "Vibration Peak"
        else:
            self.active_event = "None"
            
        # 3. Base Noise Sampling (Independent stochastic component per sensor)
        raw_temp = temp_mean + np.random.normal(0, max(0.1, sigma_t)) + (3.2 * spike_factor)
        raw_vib = vib_mean + np.random.normal(0, max(0.01, sigma_v)) + (0.45 * spike_factor)
        
        # 4. Correlated Multi-Sensor Coupling (Physical engineering cross-coupling)
        # - Temperature rise drives additional motor current demand
        # - Higher mechanical vibration generates elevated acoustic noise
        curr_thermal_coupling = 0.12 * max(0.0, raw_temp - self.temp_base)
        noise_vib_coupling = 3.8 * max(0.0, raw_vib - self.vib_base)
        
        raw_curr = curr_mean + curr_thermal_coupling + np.random.normal(0, max(0.02, sigma_c)) + (1.4 * spike_factor)
        raw_noise = noise_mean + noise_vib_coupling + np.random.normal(0, max(0.2, sigma_n)) + (4.5 * spike_factor)
        
        temp_measured = max(20.0, raw_temp)
        vib_measured = max(0.05, raw_vib)
        curr_measured = max(4.0, raw_curr)
        noise_measured = max(30.0, raw_noise)
        
        # 5. Monotonic Physical Cumulative Wear & Descriptive Health Accumulation
        d_temp = max(0.0, (temp_measured - self.temp_base) / 15.0) ** 1.3
        d_vib = max(0.0, (vib_measured - self.vib_base) / 4.0) ** 1.3
        d_curr = max(0.0, (curr_measured - self.curr_base) / 7.2) ** 1.3
        d_noise = max(0.0, (noise_measured - self.noise_base) / 30.0) ** 1.3
        
        current_raw_wear = 0.30 * d_temp + 0.35 * d_vib + 0.20 * d_curr + 0.15 * d_noise
        current_raw_wear = max(current_raw_wear, p * 0.95)
        
        self.cum_wear = max(self.cum_wear, current_raw_wear)
        
        # Descriptive continuous health decay throughout healthy and wear phases
        age_health_decay = (day / float(self.target_lifespan)) * 12.0
        wear_health_decay = self.cum_wear * 87.0
        health_calc = 100.0 - (age_health_decay + wear_health_decay)

        # MONOTONIC HEALTH DECAY: Health may only decrease or stay constant
        health_clamped = max(0.0, min(100.0, round(float(health_calc), 1)))
        self.health = min(self.prev_health, health_clamped)
        if self.prev_health == 0.0:
            self.health = 0.0
        self.prev_health = self.health
        
        # 4. Ground-Truth Remaining Useful Life (RUL)
        self.rul = max(0, int(self.target_lifespan - day))

        
        timestamp_str = (self.start_time + datetime.timedelta(days=self.current_step)).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "Timestamp": timestamp_str,
            "Temperature": round(float(temp_measured), 2),
            "Vibration": round(float(vib_measured), 2),
            "Motor_Current": round(float(curr_measured), 2),
            "Acoustic_Noise": round(float(noise_measured), 2),
            "Machine_Health": round(float(self.health), 1),
            "Machine_Status": self._evaluate_stage(),
            "Remaining_Useful_Life_Days": int(self.rul),
            "Active_Event": self.active_event,
            "Day": int(self.current_step)
        }

# Alias for backwards compatibility
MachineSimulator = RealTimeMachineSimulator
