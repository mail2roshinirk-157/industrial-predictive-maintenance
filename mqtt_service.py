"""MQTT Telemetry Ingestion Service.

Provides asynchronous, optional telemetry ingestion via MQTT for a single machine.
Ensures zero-crash behavior if MQTT broker is absent or disconnects.
"""

import json
import logging
import threading
import time
from typing import Callable, Optional, Dict, Any

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from config import (
    FEATURE_COLUMNS,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    MQTT_MACHINE_ID,
    MQTT_KEEPALIVE
)

logger = logging.getLogger("MQTTTelemetryService")


class MQTTTelemetryService:
    def __init__(
        self,
        broker: str = MQTT_BROKER,
        port: int = MQTT_PORT,
        topic: str = MQTT_TOPIC,
        machine_id: str = MQTT_MACHINE_ID,
        keepalive: int = MQTT_KEEPALIVE
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.machine_id = machine_id
        self.keepalive = keepalive
        
        self.client: Optional[Any] = None
        self.is_connected = False
        self._running = False
        self._on_reading_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._thread: Optional[threading.Thread] = None
        
        # Ingestion metrics
        self.messages_received = 0
        self.messages_parsed = 0
        self.messages_dropped = 0
        self.last_payload_timestamp: Optional[str] = None
        self.connection_errors = 0

    def start(self, on_reading: Callable[[Dict[str, Any]], None]) -> bool:
        """Start the background MQTT client loop."""
        if not PAHO_AVAILABLE:
            logger.error("paho-mqtt is not installed. MQTT service cannot start.")
            return False

        self._on_reading_callback = on_reading
        self._running = True

        # Initialize paho client safely supporting v1 and v2 API
        try:
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"industrial_backend_{self.machine_id}"
                )
            else:
                self.client = mqtt.Client(
                    client_id=f"industrial_backend_{self.machine_id}"
                )
        except Exception as e:
            logger.error(f"Failed to create MQTT client: {e}")
            return False

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Start async worker thread for connection and reconnection management
        self._thread = threading.Thread(target=self._connection_worker, daemon=True)
        self._thread.start()
        return True

    def _connection_worker(self):
        """Worker thread to connect and keep client running without blocking backend."""
        logger.info(f"MQTT service attempting connection to {self.broker}:{self.port} for topic '{self.topic}'...")
        while self._running:
            try:
                if not self.is_connected and self.client:
                    self.client.connect(self.broker, self.port, keepalive=self.keepalive)
                    self.client.loop_start()
                    # Wait while loop is active or until disconnected
                    while self._running and self.is_connected:
                        time.sleep(1.0)
            except Exception as e:
                self.connection_errors += 1
                logger.warning(
                    f"MQTT broker connection unavailable at {self.broker}:{self.port} ({e}). "
                    f"Retrying in 5 seconds..."
                )
                time.sleep(5.0)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for connection established."""
        rc_code = getattr(rc, "value", rc)
        if rc_code == 0:
            self.is_connected = True
            logger.info(f"MQTT connected to {self.broker}:{self.port}. Subscribing to '{self.topic}'...")
            client.subscribe(self.topic)
        else:
            self.is_connected = False
            logger.warning(f"MQTT connection refused with result code: {rc}")

    def _on_disconnect(self, client, userdata, rc, *args, **kwargs):
        """Callback for disconnection."""
        self.is_connected = False
        logger.warning(f"MQTT disconnected (rc={rc}). Waiting for auto-reconnect...")

    def _on_message(self, client, userdata, msg):
        """Process incoming telemetry message."""
        self.messages_received += 1
        try:
            payload_str = msg.payload.decode("utf-8")
            data = json.loads(payload_str)
            normalized = self._normalize_payload(data)
            if normalized:
                self.messages_parsed += 1
                self.last_payload_timestamp = normalized.get("Timestamp")
                if self._on_reading_callback:
                    self._on_reading_callback(normalized)
            else:
                self.messages_dropped += 1
        except Exception as e:
            self.messages_dropped += 1
            logger.error(f"Error handling MQTT message on topic '{msg.topic}': {e}")

    def _normalize_payload(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and normalize payload against the 10 canonical features."""
        # Check machine_id filter
        m_id = data.get("Machine_ID", self.machine_id)
        if m_id != self.machine_id:
            # Skip messages meant for other machines
            return None

        # Build normalized dictionary
        normalized: Dict[str, Any] = {
            "Machine_ID": m_id,
            "Machine_Name": data.get("Machine_Name", f"Machine {m_id}"),
            "Timestamp": data.get("Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "Tick": int(data.get("Tick", 0)),
            "Day": int(data.get("Day", 0)),
            "Active_Event": data.get("Active_Event", "Normal Operation")
        }

        # Check all 10 canonical features
        for feat in FEATURE_COLUMNS:
            if feat not in data:
                logger.warning(f"MQTT payload missing canonical feature '{feat}'. Dropping message.")
                return None
            try:
                normalized[feat] = float(data[feat])
            except (ValueError, TypeError):
                logger.warning(f"MQTT feature '{feat}' is not numeric: {data[feat]}. Dropping message.")
                return None

        return normalized

    def stop(self):
        """Stop MQTT client."""
        self._running = False
        self.is_connected = False
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting client: {e}")
        logger.info("MQTT service stopped.")

    def get_stats(self) -> Dict[str, Any]:
        """Return operational telemetry statistics."""
        return {
            "broker": self.broker,
            "port": self.port,
            "topic": self.topic,
            "machine_id": self.machine_id,
            "is_connected": self.is_connected,
            "messages_received": self.messages_received,
            "messages_parsed": self.messages_parsed,
            "messages_dropped": self.messages_dropped,
            "connection_errors": self.connection_errors,
            "last_payload_timestamp": self.last_payload_timestamp
        }
