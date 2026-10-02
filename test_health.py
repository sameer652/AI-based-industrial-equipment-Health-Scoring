"""Tests for health classification, scoring, alerts and API validation."""
import pytest
from fastapi.testclient import TestClient

import api

HEALTHY = {"equipment_id": 3, "temperature": 62, "vibration": 2.1,
           "pressure": 99, "rpm": 1495, "current": 10.1}
WORN = {"equipment_id": 3, "temperature": 84, "vibration": 7.8,
        "pressure": 81, "rpm": 1360, "current": 13.8}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Each test gets its own empty database, so the real one is untouched."""
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "test.db"))
    api.init_db()
    return TestClient(api.app)


@pytest.mark.parametrize("score,expected", [
    (100, "HEALTHY"), (80, "HEALTHY"), (79.9, "WARNING"),
    (50, "WARNING"), (49.9, "CRITICAL"), (0, "CRITICAL"),
])
def test_classify_thresholds(score, expected):
    assert api.classify(score) == expected


def test_model_scores_healthy_high_and_worn_low():
    assert api.calculate_health(HEALTHY) >= 80
    assert api.calculate_health(WORN) < 50


def test_healthy_reading_creates_no_alert(client):
    r = client.post("/readings", json=HEALTHY)
    assert r.status_code == 200
    assert r.json()["status"] == "HEALTHY"
    assert r.json()["alert_created"] is False


def test_worn_reading_creates_alert_once(client):
    first = client.post("/readings", json=WORN).json()
    second = client.post("/readings", json=WORN).json()
    assert first["status"] == "CRITICAL"
    assert first["alert_created"] is True
    assert second["alert_created"] is False          # no alert spam
    assert len(client.get("/alerts").json()) == 1


def test_invalid_reading_rejected(client):
    bad = {**HEALTHY, "temperature": -5}
    assert client.post("/readings", json=bad).status_code == 422


def test_unknown_equipment_returns_404(client):
    assert client.post("/readings", json={**HEALTHY, "equipment_id": 99}).status_code == 404


def test_equipment_list_has_20_machines(client):
    assert len(client.get("/equipment").json()) == 20


def test_acknowledge_alert(client):
    client.post("/readings", json=WORN)
    alert_id = client.get("/alerts").json()[0]["id"]
    assert client.post(f"/alerts/{alert_id}/acknowledge").status_code == 200
    assert client.get("/alerts").json() == []