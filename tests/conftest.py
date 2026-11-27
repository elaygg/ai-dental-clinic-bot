"""Shared test setup. pytest loads this file by itself, before any test file.

The tests never call OpenAI: they need no real key, no internet, and cost nothing.
Inside each test, the LLM and the vector search are replaced with fakes
(see monkeypatch in test_assistant.py).
"""

import os

import pytest
from fastapi.testclient import TestClient

# Fixed settings for every test run. Environment variables win over the .env
# file, so the tests behave the same on your Mac and in CI, whatever is in .env.
# They must be set before the app is imported: app/main.py reads settings on import.
TEST_API_KEY = "test-api-key"
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["API_KEY"] = TEST_API_KEY
# Safety net: if a test forgets to replace the LLM, the call goes to a closed
# local port and fails right away, instead of going to the internet.
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"


@pytest.fixture
def client():
    """Sends requests straight into the FastAPI app: no server, no network."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    # Undo any dependency overrides a test made, so the next test starts clean.
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """The header a real client (Make.com) sends with every request."""
    return {"X-API-Key": TEST_API_KEY}
