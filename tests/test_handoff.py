"""Level 1: plain functions from app/handoff.py.

No fakes needed here: these functions do not call the LLM,
so the same input always gives the same output.
"""

import pytest

from app.config import get_settings
from app.handoff import HandoffReason, asks_for_human, detect_language, handoff_message


# parametrize runs the same test once per value in the list.
# In the output you will see 5 separate tests, one per message.
@pytest.mark.parametrize(
    "message",
    [
        "Can I talk to a human?",
        "I want to speak with the administrator",
        "Please call me back",
        "Позовите администратора",
        "Хочу поговорить с живым человеком",
    ],
)
def test_asks_for_human_catches_explicit_requests(message):
    assert asks_for_human(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "How much is a kids' dental cleaning?",
        "Сколько стоит чистка?",
        "Is the administration fee included?",
    ],
)
def test_asks_for_human_ignores_normal_questions(message):
    assert asks_for_human(message) is False


@pytest.mark.parametrize(
    "text, expected",
    [("Привет", "ru"), ("Hello", "en"), ("Hello, Привет", "ru"), ("123", "en")],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_medical_message_includes_clinic_phone():
    message = handoff_message(HandoffReason.MEDICAL_QUESTION, "My tooth hurts")
    assert get_settings().clinic_phone in message


def test_handoff_message_answers_in_russian():
    message = handoff_message(HandoffReason.MEDICAL_QUESTION, "Болит зуб")
    assert "врач" in message
