"""Level 2: the decision flow in reply_to(), with the LLM and the search replaced by fakes.

monkeypatch.setattr("module.name", fake) swaps a function for one test only,
and puts the original back when the test ends.

We patch the names inside app.assistant, not inside app.handoff or app.rag:
app/assistant.py did `from app.handoff import classify`, so it holds its own
reference to the function, and reply_to() calls that reference.
Rule of thumb: patch where the function is used, not where it is defined.
"""

from app.assistant import GREETING, reply_to
from app.handoff import HandoffReason
from app.rag import RagAnswer


def must_not_be_called(*args, **kwargs):
    """A fake that fails the test if the code under test calls it."""
    raise AssertionError("this function must not be called here")


def test_explicit_request_for_human_skips_the_llm(monkeypatch):
    # Arrange: if the keyword rules work, neither the classifier nor RAG is reached.
    monkeypatch.setattr("app.assistant.classify", must_not_be_called)
    monkeypatch.setattr("app.assistant.answer_question", must_not_be_called)

    # Act
    reply = reply_to("Can I talk to an administrator?")

    # Assert
    assert reply.handoff is True
    assert reply.handoff_reason == HandoffReason.HUMAN_REQUESTED
    assert reply.sources == []


def test_medical_question_goes_to_a_person(monkeypatch):
    monkeypatch.setattr("app.assistant.classify", lambda text: "medical_advice")
    monkeypatch.setattr("app.assistant.answer_question", must_not_be_called)

    reply = reply_to("My gum is swollen, what should I do?")

    assert reply.handoff_reason == HandoffReason.MEDICAL_QUESTION


def test_yes_is_a_booking_request(monkeypatch):
    monkeypatch.setattr("app.assistant.classify", lambda text: "booking_request")
    monkeypatch.setattr("app.assistant.answer_question", must_not_be_called)

    reply = reply_to("Yes")

    assert reply.handoff_reason == HandoffReason.BOOKING_REQUESTED


def test_small_talk_gets_greeting_in_russian(monkeypatch):
    monkeypatch.setattr("app.assistant.classify", lambda text: "small_talk")
    monkeypatch.setattr("app.assistant.answer_question", must_not_be_called)

    reply = reply_to("Привет")

    assert reply.answer == GREETING["ru"]
    assert reply.handoff is False


def test_nothing_found_goes_to_a_person(monkeypatch):
    monkeypatch.setattr("app.assistant.classify", lambda text: "clinic_info")
    monkeypatch.setattr(
        "app.assistant.answer_question", lambda question: RagAnswer(answer=None, sources=[], top_score=0.1)
    )

    reply = reply_to("Do you have a parking?")

    assert reply.handoff_reason == HandoffReason.NO_RELEVANT_INFO


def test_clinic_question_gets_rag_answer(monkeypatch):
    monkeypatch.setattr("app.assistant.classify", lambda text: "clinic_info")
    monkeypatch.setattr(
        "app.assistant.answer_question",
        lambda question: RagAnswer(
            answer="A kids' cleaning costs 17,000 ₸.", sources=["apostol_clinic_eng.md"], top_score=0.7
        ),
    )

    reply = reply_to("How much is a kid cleaning?")

    assert reply.handoff is False
    assert reply.answer == "A kids' cleaning costs 17,000 ₸."
    assert reply.sources == ["apostol_clinic_eng.md"]
