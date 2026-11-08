"""HTTP-level tests for mevin.py: auth, path safety, internal endpoints, alert relay.

Needs fastapi + httpx (installed in CI). No camera, GPU or Ollama required:
the default camera simply never connects.
"""
import os
import sys
import tempfile
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "test-secret-token"


@pytest.fixture(scope="module")
def mevin():
    tmp = tempfile.mkdtemp(prefix="mevin-test-")
    os.environ.update({
        "MEVIN_DB": os.path.join(tmp, "monitor.db"),
        "MEVIN_SNAPSHOTS": os.path.join(tmp, "snapshots"),
        "MEVIN_TOKEN": TOKEN,
        "MEVIN_INTERNAL_KEY": "internal-key",
    })
    sys.path.insert(0, ROOT)
    import mevin as m
    yield m
    m.analyzer.stop()


@pytest.fixture(scope="module")
def client(mevin):
    from fastapi.testclient import TestClient
    return TestClient(mevin.app)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_health_and_dashboard_are_public(client):
    assert client.get("/healthz").json()["ok"] is True
    assert client.get("/").status_code == 200


def test_api_requires_token(client):
    assert client.get("/api/settings").status_code == 401
    assert client.get("/api/settings", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/settings", headers=AUTH).status_code == 200
    assert client.get(f"/api/settings?token={TOKEN}").status_code == 200


def test_cookie_auth_for_streams(client):
    # Live video / SSE / snapshots can't send headers, so the dashboard uses a cookie
    assert client.get("/video/cam0").status_code == 401
    assert client.get("/snapshot-live/cam0").status_code == 401
    assert client.get("/events").status_code == 401
    r = client.get("/snapshot-live/cam0", cookies={"mevin_token": TOKEN})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_snapshot_path_traversal_blocked(client, mevin):
    day = os.path.join(mevin.SNAPSHOT_DIR, "2026-01-01")
    os.makedirs(day, exist_ok=True)
    with open(os.path.join(day, "cam0_120000.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")
    assert client.get("/snapshot/2026-01-01/cam0_120000.jpg", headers=AUTH).status_code == 200
    assert client.get("/snapshot//etc/passwd", headers=AUTH).status_code in (403, 404)
    assert client.get("/snapshot/%2E%2E/monitor.db", headers=AUTH).status_code in (403, 404)


def test_internal_endpoints_reject_non_local(client):
    # TestClient's peer address is not loopback, so even the right key is refused
    r = client.get("/internal/cameras", headers={"X-Mevin-Internal": "internal-key"})
    assert r.status_code == 403
    assert client.post("/internal/event", json={"type": "feed_item"}).status_code == 403


def test_settings_cannot_overwrite_internal_keys(client, mevin):
    client.post("/api/settings", headers=AUTH, json={"_worker_heartbeat": "9999999999", "max_tokens": "60"})
    s = client.get("/api/settings", headers=AUTH).json()
    assert s["max_tokens"] == "60"
    assert s.get("_worker_heartbeat", "") != "9999999999"


def test_worker_alerts_reach_the_phone(client, mevin, monkeypatch):
    """Regression: with the separate worker process, alerts were never pushed."""
    sent = []
    monkeypatch.setattr(mevin, "send_push", lambda text, photo=None, is_alert=True: sent.append((text, is_alert)))
    mevin.app.dependency_overrides[mevin.check_internal] = lambda: None
    try:
        r = client.post("/internal/event", json={
            "type": "feed_item", "item_id": "cam0_1", "camera_id": "cam0", "camera_name": "Front door",
            "time": "12:00:00", "text": "A man is forcing the door.", "is_alert": True,
            "alert_keywords": ["forcing"], "no_change": False, "_save": True,
        })
        assert r.status_code == 200
        for _ in range(50):
            if sent:
                break
            time.sleep(0.02)
        assert sent and sent[0][1] is True and "Front door" in sent[0][0]
    finally:
        mevin.app.dependency_overrides.clear()
