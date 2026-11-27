"""Level 3: the HTTP API, called through TestClient.

The `client` and `auth_headers` arguments are fixtures from conftest.py:
pytest sees the argument name and passes in what the fixture returns.
"""

import pytest
from pydantic import SecretStr

from app.assistant import AssistantReply
from app.config import Settings, get_settings
from app.handoff import HandoffReason
from app.rag import IndexNotReadyError


def must_not_be_called(*args, **kwargs):
    """A fake that fails the test if the code under test calls it."""
    raise AssertionError("this function must not be called here")


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_chat_requires_api_key(client, monkeypatch, headers):
    monkeypatch.setattr("app.main.reply_to", must_not_be_called)

    response = client.post("/chat", json={"question": "Hi"}, headers=headers)

    assert response.status_code == 401


@pytest.mark.parametrize("body", [{"question": " "}, {}])
def test_chat_rejects_empty_question(client, auth_headers, monkeypatch, body):
    monkeypatch.setattr("app.main.reply_to", must_not_be_called)

    response = client.post("/chat", json=body, headers=auth_headers)

    assert response.status_code == 422


def test_chat_returns_handoff_fields(client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        "app.main.reply_to",
        lambda question: AssistantReply(
            answer="Passed to admin", handoff_reason=HandoffReason.MEDICAL_QUESTION
        ),
    )

    response = client.post("/chat", json={"question": "My tooth hurts"}, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["handoff"] is True
    assert data["handoff_reason"] == "medical_question"
    assert data["sources"] == []


@pytest.mark.parametrize("path", ["/chat", "/ask"])
def test_both_endpoints_answer(client, auth_headers, monkeypatch, path):
    monkeypatch.setattr("app.main.reply_to", lambda question: AssistantReply(answer="Hello"))

    response = client.post(path, json={"question": "Hi"}, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["answer"] == "Hello"


def test_openai_failure_returns_502(client, auth_headers, monkeypatch):
    def broken_reply_to(question):
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr("app.main.reply_to", broken_reply_to)

    response = client.post("/chat", json={"question": "Hi"}, headers=auth_headers)

    assert response.status_code == 502


def test_missing_index_returns_503(client, auth_headers, monkeypatch):
    def broken_reply_to(question):
        raise IndexNotReadyError("no index")

    monkeypatch.setattr("app.main.reply_to", broken_reply_to)

    response = client.post("/chat", json={"question": "Hi"}, headers=auth_headers)

    assert response.status_code == 503


def test_ask_is_marked_deprecated(client):
    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/ask"]["post"]["deprecated"] is True


def test_no_api_key_configured_means_no_auth(client, monkeypatch):
    settings_without_key = Settings(openai_api_key=SecretStr("sk-test"), api_key=None)
    client.app.dependency_overrides[get_settings] = lambda: settings_without_key
    monkeypatch.setattr("app.main.reply_to", lambda question: AssistantReply(answer="Hello"))

    response = client.post("/chat", json={"question": "Hi"})

    assert response.status_code == 200
