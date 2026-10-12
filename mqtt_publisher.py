"""Standalone MQTT Telemetry Publisher.

Publishes single-machine industrial telemetry conforming to the 10-feature schema
to a local or remote MQTT broker (default: localhost:1883).
"""

import argparse
import json
import logging
import random
import time
from typing import Dict, Any

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

from config import (
    FEATURE_COLUMNS,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_MACHINE_ID
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MQTTPublisher")


def generate_machine_telemetry(machine_id: str, tick: int, wear_progress: float = 0.15) -> Dict[str, Any]:
    """
    Generates a realistic telemetry payload with the 10 canonical v2 features,
    maintaining consistency with physics and simulator dynamics.
    """
    # AR(1) style Load_Percent around 70%
    load_pct = float(np_clip := max(40.0, min(100.0, random.gauss(70.0, 8.0))))
    
    # Base operational sensors with realistic load coupling
    temp = round(60.0 + 8.0 * (load_pct / 100.0) + 12.0 * wear_progress + random.gauss(0, 0.5), 2)
    vib = round(0.18 + 0.05 * (load_pct / 100.0) + 0.8 * (wear_progress ** 2) + random.gauss(0, 0.02), 3)
    curr = round(7.5 + 2.5 * (load_pct / 100.0) + 2.0 * wear_progress + random.gauss(0, 0.1), 2)
    press = round(5.0 + 0.8 * (load_pct / 100.0) - 0.4 * wear_progress + random.gauss(0, 0.05), 2)
    rpm = round(1750.0 - 20.0 * (load_pct / 100.0) - 30.0 * wear_progress + random.gauss(0, 5.0), 1)
    flow = round(150.0 + 10.0 * (load_pct / 100.0) - 15.0 * wear_progress + random.gauss(0, 1.5), 1)
    oil_temp = round(45.0 + 7.0 * (load_pct / 100.0) + 8.0 * wear_progress + random.gauss(0, 0.4), 2)
    pwr = round(15.0 + 6.0 * (load_pct / 100.0) + 3.0 * wear_progress + random.gauss(0, 0.2), 2)
    
    # Oil debris count (Poisson-like degradation)
    debris_mean = 5.0 + 400.0 * (wear_progress ** 3)
    debris_count = max(0, int(round(random.gauss(debris_mean, max(1.0, debris_mean * 0.1)))))

    health = max(0.0, min(100.0, round(100.0 * (1.0 - wear_progress), 1)))

    return {
        "Machine_ID": machine_id,
        "Machine_Name": f"Machine {machine_id}",
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Tick": tick,
        "Day": max(1, tick // 24),
        "Machine_Health": health,
        "Active_Event": "Normal Operation" if wear_progress < 0.6 else "High Vibration",
        "Temperature": temp,
        "Vibration": vib,
        "Motor_Current": curr,
        "Pressure": press,
        "RPM": rpm,
        "Flow_Rate": flow,
        "Oil_Temperature": oil_temp,
        "Power_Consumption": pwr,
        "Load_Percent": round(load_pct, 2),
        "Oil_Debris_Count": debris_count
    }


def main():
    parser = argparse.ArgumentParser(description="Publish single-machine industrial telemetry over MQTT.")
    parser.add_argument("--broker", type=str, default=MQTT_BROKER, help="MQTT Broker host")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help="MQTT Broker port")
    parser.add_argument("--topic", type=str, default=MQTT_TOPIC, help="MQTT Topic")
    parser.add_argument("--machine", type=str, default=MQTT_MACHINE_ID, help="Machine ID")
    parser.add_argument("--interval", type=float, default=1.0, help="Publish interval in seconds")
    parser.add_argument("--ticks", type=int, default=0, help="Total ticks to publish (0 = infinite)")
    args = parser.parse_args()

    if mqtt is None:
        logger.error("paho-mqtt library is not installed. Please install it with 'pip install paho-mqtt'.")
        return

    logger.info(f"Connecting to MQTT Broker at {args.broker}:{args.port}...")
    try:
        if hasattr(mqtt, "CallbackAPIVersion"):
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim_publisher_{args.machine}")
        else:
            client = mqtt.Client(client_id=f"sim_publisher_{args.machine}")
        client.connect(args.broker, args.port, keepalive=60)
        client.loop_start()
    except Exception as e:
        logger.error(f"Could not connect to MQTT Broker: {e}")
        return

    logger.info(f"Connected! Publishing to topic '{args.topic}' for machine {args.machine} (interval: {args.interval}s)...")
    
    tick = 1
    wear_progress = 0.15
    try:
        while True:
            # Gradually increase wear slightly over time
            wear_progress = min(0.95, wear_progress + 0.001)
            telemetry = generate_machine_telemetry(args.machine, tick, wear_progress)
            payload = json.dumps(telemetry)
            
            res = client.publish(args.topic, payload)
            if res.rc == 0:
                logger.info(
                    f"Tick {tick:04d} -> {args.topic} | Load: {telemetry['Load_Percent']:.1f}% | "
                    f"Debris: {telemetry['Oil_Debris_Count']} | Temp: {telemetry['Temperature']} C | "
                    f"Vib: {telemetry['Vibration']} mm/s"
                )
            else:
                logger.warning(f"Tick {tick:04d} publish failed with return code {res.rc}")

            if args.ticks > 0 and tick >= args.ticks:
                logger.info(f"Completed requested {args.ticks} ticks. Exiting.")
                break

            tick += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Publisher stopped.")


if __name__ == "__main__":
    main()
