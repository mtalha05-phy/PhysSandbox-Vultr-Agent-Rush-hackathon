"""End-to-end API integration tests for FastAPI service."""

import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_and_status():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

    res_stat = client.get("/api/status")
    assert res_stat.status_code == 200
    data = res_stat.json()
    assert "experiments" in data
    # All 6 experiments must be registered
    assert len(data["experiments"]) == 6
    assert "helmholtz" in data["experiments"]
    assert "laser_profile" in data["experiments"]
    assert "faraday" in data["experiments"]
    assert "beer_lambert" in data["experiments"]


def test_samples_endpoint():
    res = client.get("/api/samples/resistance_clean.csv")
    assert res.status_code == 200
    assert "current_A,voltage_V" in res.text


def test_analyse_resistance_clean():
    csv_content = "current_A,voltage_V\n0.01,0.0001\n0.02,0.0002\n0.03,0.0003\n0.04,0.0004\n"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    data = {"experiment": "resistance_4wire", "expected": json.dumps({"resistance_ohm": 0.010})}

    res = client.post("/api/analyse", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert "values" in body
    assert "resistance_fit_ohm" in body["values"]
    assert "receipt" in body
    assert body["receipt"]["outcome"] == ["clean"]


def test_analyse_helmholtz():
    csv_content = "current_A,measured_B_uT\n0.1,60.0\n0.2,120.0\n0.3,180.0\n0.4,240.0\n"
    files = {"file": ("test_mag.csv", csv_content, "text/csv")}
    data = {"experiment": "helmholtz", "expected": json.dumps({"radius_m": 0.15, "turns": 100})}

    res = client.post("/api/analyse", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert "values" in body
    assert "k_measured_uT_per_A" in body["values"]


def test_containment_demos():
    for attack in ["egress", "runaway", "host_read"]:
        res = client.post(f"/api/containment/{attack}")
        assert res.status_code == 200
        assert res.json()["contained"] is True

    res_log = client.get("/api/containment-log")
    assert res_log.status_code == 200
    assert len(res_log.json()["events"]) > 0


def test_ledger_integrity():
    res = client.get("/api/ledger")
    assert res.status_code == 200
    body = res.json()
    assert body["verify"]["intact"] is True
    assert body["verify"]["receipts"] > 0
