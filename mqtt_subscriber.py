"""
MQTT subscriber: listens to sensor telemetry, sends each reading to the
scoring API (Random Forest + SQLite), then publishes the health score
back to MQTT on <prefix>/equipment/<id>/health.
"""
import json
import os

import paho.mqtt.client as mqtt
import requests

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PREFIX = os.getenv("MQTT_PREFIX", "factory")
API = os.getenv("API_URL", "http://127.0.0.1:8000")

SUBSCRIBE_TOPIC = f"{PREFIX}/equipment/+/telemetry"


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"MQTT connection failed: {reason_code}")
        return
    client.subscribe(SUBSCRIBE_TOPIC, qos=1)
    print(f"Connected. Subscribed to {SUBSCRIBE_TOPIC}")


def on_message(client, userdata, msg):
    try:
        reading = json.loads(msg.payload)
        resp = requests.post(f"{API}/readings", json=reading, timeout=5)
        resp.raise_for_status()
        result = resp.json()
    except json.JSONDecodeError:
        print(f"Bad JSON on {msg.topic}, skipped")
        return
    except requests.RequestException as e:
        print(f"API error for {msg.topic}: {e}")
        return

    # publish the computed health score back to MQTT
    health_topic = f"{PREFIX}/equipment/{result['equipment_id']}/health"
    client.publish(health_topic, json.dumps(result), qos=1)

    flag = "  ALERT" if result["alert_created"] else ""
    print(f"{result['equipment_name']}: {result['health_score']} ({result['status']}){flag}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="health-subscriber")
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

print(f"Connecting to MQTT broker {MQTT_HOST}:{MQTT_PORT} ...")
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Stopping subscriber...")
    client.disconnect()