"""The app starts on an empty data directory and its main pages render."""


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_home_and_logs_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/logs").status_code == 200


def test_create_project_and_open_it(client):
    r = client.post("/web/projects", data={"name": "Test site", "domain": "example.com",
                                           "ollama_model": "m"})
    assert r.status_code == 204
    url = r.headers["hx-redirect"]
    page = client.get(url)
    assert page.status_code == 200
    assert "Test site" in page.text


def test_api_key_never_reaches_the_log(client, data_dir):
    import logging
    import os
    logging.getLogger("app.test").warning("failure talking to host with key test-secret-key-0123456789")
    for h in logging.getLogger().handlers:
        h.flush()
    with open(os.path.join(data_dir, "logs", "web.log"), encoding="utf-8") as f:
        text = f.read()
    assert "test-secret-key-0123456789" not in text
    assert "***" in text


def test_password_protects_everything_but_health(client, monkeypatch):
    import base64
    monkeypatch.setenv("DA_PASSWORD", "hunter2-pass")
    assert client.get("/").status_code == 401
    assert client.get("/").headers["www-authenticate"].startswith("Basic")
    assert client.get("/api/health").status_code == 200
    bad = base64.b64encode(b"archive:wrong").decode()
    assert client.get("/", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    good = base64.b64encode(b"archive:hunter2-pass").decode()
    assert client.get("/", headers={"Authorization": f"Basic {good}"}).status_code == 200
    monkeypatch.setenv("DA_USERNAME", "me")
    assert client.get("/", headers={"Authorization": f"Basic {good}"}).status_code == 401


def test_password_is_redacted(monkeypatch):
    from app.config import get_llm_config
    monkeypatch.setenv("DA_PASSWORD", "hunter2-pass")
    assert "hunter2-pass" not in get_llm_config().redact("login failed for hunter2-pass")
