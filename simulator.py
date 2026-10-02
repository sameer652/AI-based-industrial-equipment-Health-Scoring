"""
Simulates live sensor data from 20 machines and PUBLISHES it over MQTT.
Topic: <prefix>/equipment/<id>/telemetry
"""
import json
import os
import random
import time

import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PREFIX = os.getenv("MQTT_PREFIX", "factory")

NUM_MACHINES = 20
LIFETIME_STEPS = 500
STEP_SIZE = 5          # how much machines age per tick (bigger = faster demo)
TICK_SECONDS = 3

rng = random.Random(42)

# Start machines at different ages so the dashboard has variety immediately
state = {
    i: {"t": rng.randint(0, 420), "speed": rng.uniform(0.7, 1.3)}
    for i in range(1, NUM_MACHINES + 1)
}


def make_reading(machine_id: int) -> dict:
    s = state[machine_id]
    d = min(1.0, (s["t"] / LIFETIME_STEPS) ** 2 * s["speed"])
    return {
        "equipment_id": machine_id,
        "temperature": round(max(0, 60 + 25 * d + rng.gauss(0, 2)), 2),
        "vibration": round(max(0, 2 + 6 * d + rng.gauss(0, 0.3)), 2),
        "pressure": round(max(0, 100 - 20 * d + rng.gauss(0, 2)), 2),
        "rpm": round(max(0, 1500 - 150 * d + rng.gauss(0, 20)), 2),
        "current": round(max(0, 10 + 4 * d + rng.gauss(0, 0.3)), 2),
    }


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sensor-simulator")
client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
client.loop_start()
print(f"Simulator connected to MQTT broker {MQTT_HOST}:{MQTT_PORT}. Press Ctrl+C to stop.")

tick = 0
try:
    while True:
        tick += 1
        for machine_id in range(1, NUM_MACHINES + 1):
            topic = f"{PREFIX}/equipment/{machine_id}/telemetry"
            client.publish(topic, json.dumps(make_reading(machine_id)), qos=1)

            state[machine_id]["t"] += STEP_SIZE
            if state[machine_id]["t"] >= LIFETIME_STEPS:   # serviced -> new machine
                state[machine_id]["t"] = 0
                state[machine_id]["speed"] = rng.uniform(0.7, 1.3)

        print(f"[tick {tick}] published readings for {NUM_MACHINES} machines")
        time.sleep(TICK_SECONDS)
except KeyboardInterrupt:
    print("Stopping simulator...")
finally:
    client.loop_stop()
    client.disconnect()