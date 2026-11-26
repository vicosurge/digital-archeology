"""Test setup: an empty data directory and an LLM endpoint that is never there.

Everything is set before the app is imported, because the database URL is
fixed at import time. No test touches the network: the LLM host is a closed
local port and the external lookups are tested with httpx.MockTransport.
"""
import os
import sys
import tempfile

_DATA = tempfile.mkdtemp(prefix="da-test-")
os.environ["DA_DATA_DIR"] = _DATA
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
os.environ["OLLAMA_API_KEY"] = "test-secret-key-0123456789"
os.environ.pop("DA_PASSWORD", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_dir():
    return _DATA


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
