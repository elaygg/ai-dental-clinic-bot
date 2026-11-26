"""The assistant's decision flow for one patient message.

    message
      -> asks for a human (keyword rules)?     -> handoff: human_requested
      -> LLM classifier, by intent:
           medical_advice                      -> handoff: medical_question
           human_request                       -> handoff: human_requested
           booking_request                     -> handoff: booking_requested
           small_talk                          -> greeting, no handoff
           clinic_info                         -> RAG below
      -> RAG: nothing relevant found?          -> handoff: no_relevant_info
      -> RAG answer from clinic data

Cheap checks run first, so most handoffs cost nothing, and medical
questions never reach the answer generator.
"""

import logging
from dataclasses import dataclass, field

from app.handoff import (
    INTENT_HANDOFF,
    HandoffReason,
    asks_for_human,
    classify,
    detect_language,
    handoff_message,
)
from app.rag import answer_question

logger = logging.getLogger(__name__)

# Reply to greetings, thanks, /start and off-topic messages. Neutral wording,
# so it fits both "Hi" and "Thank you".
GREETING = {
    "en": (
        "Happy to help! I'm the virtual assistant of Apostol Dental Clinic: ask me about prices, "
        'services, doctors or payment options. To talk to a person, just write "administrator".'
    ),
    "ru": (
        "Рад помочь! Я виртуальный помощник Apostol Dental Clinic: спросите меня о ценах, "
        "услугах, врачах или способах оплаты. Чтобы поговорить с человеком, напишите «администратор»."
    ),
}


@dataclass
class AssistantReply:
    answer: str
    sources: list[str] = field(default_factory=list)
    handoff_reason: HandoffReason | None = None

    @property
    def handoff(self) -> bool:
        return self.handoff_reason is not None


def _handoff(reason: HandoffReason, message: str) -> AssistantReply:
    logger.info("handoff reason=%s", reason)
    return AssistantReply(answer=handoff_message(reason, message), handoff_reason=reason)


def reply_to(message: str) -> AssistantReply:
    # 1. Explicit request for a person: no LLM call needed.
    if asks_for_human(message):
        return _handoff(HandoffReason.HUMAN_REQUESTED, message)

    # 2. What does the patient want?
    intent = classify(message)
    if intent in INTENT_HANDOFF:
        return _handoff(INTENT_HANDOFF[intent], message)
    if intent == "small_talk":
        return AssistantReply(answer=GREETING[detect_language(message)])

    # 3. Question about the clinic: answer from clinic data, or hand off if nothing relevant.
    rag = answer_question(message)
    if rag.answer is None:
        return _handoff(HandoffReason.NO_RELEVANT_INFO, message)

    return AssistantReply(answer=rag.answer, sources=rag.sources)
