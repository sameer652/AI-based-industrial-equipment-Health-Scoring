"""
Equipment Health Scoring API (FastAPI + SQLite).
Receives sensor readings, scores machine health with the trained model,
stores everything, and raises alerts when health drops.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = "data/health.db"
MODEL_PATH = "models/health_model.joblib"
METRICS_PATH = "models/metrics.json"

MACHINE_TYPES = ["Motor", "Pump", "Compressor", "Turbine"]
LOCATIONS = ["Plant A", "Plant B", "Plant C"]
NUM_MACHINES = 20

# ---------- Model ----------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]


def classify(score: float) -> str:
    """Convert a 0-100 health score into a status label."""
    if score >= 80:
        return "HEALTHY"
    if score >= 50:
        return "WARNING"
    return "CRITICAL"


def calculate_health(reading: dict) -> float:
    row = pd.DataFrame([[reading[f] for f in FEATURES]], columns=FEATURES)
    failure_prob = model.predict_proba(row)[0][1]
    return round((1 - failure_prob) * 100, 1)


# ---------- Database ----------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS equipment (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                location TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL REFERENCES equipment(id),
                timestamp TEXT NOT NULL,
                temperature REAL, vibration REAL, pressure REAL,
                rpm REAL, current REAL
            );
            CREATE TABLE IF NOT EXISTS health_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL REFERENCES equipment(id),
                reading_id INTEGER NOT NULL REFERENCES sensor_readings(id),
                score REAL NOT NULL,
                status TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                equipment_id INTEGER NOT NULL REFERENCES equipment(id),
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                created_at TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_readings_equipment
                ON sensor_readings(equipment_id);
            CREATE INDEX IF NOT EXISTS idx_scores_equipment
                ON health_scores(equipment_id);
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM equipment").fetchone()[0]
        if count == 0:
            for i in range(1, NUM_MACHINES + 1):
                mtype = MACHINE_TYPES[i % len(MACHINE_TYPES)]
                conn.execute(
                    "INSERT INTO equipment (id, name, type, location) VALUES (?,?,?,?)",
                    (i, f"{mtype}-{i:02d}", mtype, LOCATIONS[i % len(LOCATIONS)]),
                )


init_db()

# ---------- API ----------
app = FastAPI(
    title="Equipment Health Scoring API",
    description="AI-based industrial equipment health scoring",
    version="1.0.0",
)


class SensorReading(BaseModel):
    equipment_id: int = Field(ge=1)
    temperature: float = Field(ge=0, le=200)
    vibration: float = Field(ge=0, le=50)
    pressure: float = Field(ge=0, le=500)
    rpm: float = Field(ge=0, le=10000)
    current: float = Field(ge=0, le=100)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.get("/")
def root():
    return {"service": "Equipment Health Scoring API", "status": "running"}


@app.post("/readings")
def submit_reading(reading: SensorReading):
    data = reading.model_dump()
    score = calculate_health(data)
    status = classify(score)

    with get_conn() as conn:
        equip = conn.execute(
            "SELECT * FROM equipment WHERE id = ?", (data["equipment_id"],)
        ).fetchone()
        if equip is None:
            raise HTTPException(404, f"Equipment {data['equipment_id']} not found")

        previous = conn.execute(
            "SELECT status FROM health_scores WHERE equipment_id = ? ORDER BY id DESC LIMIT 1",
            (data["equipment_id"],),
        ).fetchone()

        ts = now()
        cur = conn.execute(
            """INSERT INTO sensor_readings
               (equipment_id, timestamp, temperature, vibration, pressure, rpm, current)
               VALUES (?,?,?,?,?,?,?)""",
            (data["equipment_id"], ts, data["temperature"], data["vibration"],
             data["pressure"], data["rpm"], data["current"]),
        )
        reading_id = cur.lastrowid
        conn.execute(
            """INSERT INTO health_scores
               (equipment_id, reading_id, score, status, calculated_at)
               VALUES (?,?,?,?,?)""",
            (data["equipment_id"], reading_id, score, status, ts),
        )

        # Raise an alert only when status CHANGES to a bad one (avoids alert spam)
        alert_created = False
        if status != "HEALTHY" and (previous is None or previous["status"] != status):
            conn.execute(
                "INSERT INTO alerts (equipment_id, message, severity, created_at) VALUES (?,?,?,?)",
                (data["equipment_id"],
                 f"{equip['name']} health dropped to {score} ({status})",
                 status, ts),
            )
            alert_created = True

    return {
        "equipment_id": data["equipment_id"],
        "equipment_name": equip["name"],
        "health_score": score,
        "status": status,
        "alert_created": alert_created,
    }


@app.get("/equipment")
def list_equipment():
    """All machines with their latest health score."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT e.id, e.name, e.type, e.location,
                      h.score, h.status, h.calculated_at
               FROM equipment e
               LEFT JOIN health_scores h
                 ON h.id = (SELECT MAX(id) FROM health_scores WHERE equipment_id = e.id)
               ORDER BY e.id"""
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/equipment/{equipment_id}/health")
def equipment_health(equipment_id: int):
    with get_conn() as conn:
        row = conn.execute(
            """SELECT e.id, e.name, e.type, e.location,
                      h.score, h.status, h.calculated_at
               FROM equipment e
               LEFT JOIN health_scores h
                 ON h.id = (SELECT MAX(id) FROM health_scores WHERE equipment_id = e.id)
               WHERE e.id = ?""",
            (equipment_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Equipment not found")
    return dict(row)


@app.get("/equipment/{equipment_id}/history")
def equipment_history(equipment_id: int, limit: int = 200):
    """Recent scores and sensor values, oldest first (ready for charts)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT r.timestamp, r.temperature, r.vibration, r.pressure,
                      r.rpm, r.current, h.score, h.status
               FROM health_scores h
               JOIN sensor_readings r ON r.id = h.reading_id
               WHERE h.equipment_id = ?
               ORDER BY h.id DESC LIMIT ?""",
            (equipment_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


@app.get("/alerts")
def list_alerts(only_open: bool = True):
    query = """SELECT a.id, a.equipment_id, e.name AS equipment_name,
                      a.message, a.severity, a.created_at, a.acknowledged
               FROM alerts a JOIN equipment e ON e.id = a.equipment_id"""
    if only_open:
        query += " WHERE a.acknowledged = 0"
    query += " ORDER BY a.id DESC"
    with get_conn() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int):
    with get_conn() as conn:
        cur = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Alert not found")
    return {"id": alert_id, "acknowledged": True}


@app.get("/model/metrics")
def model_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)