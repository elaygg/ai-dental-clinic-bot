"""The assistant's decision flow for one patient message.

    message
      -> asks for a human (keyword rules)?  -> handoff: human_requested
      -> LLM classifier: medical / human?   -> handoff: medical_question / human_requested
      -> RAG: nothing relevant found?       -> handoff: no_relevant_info
      -> RAG answer from clinic data

Cheap checks run first, so most handoffs cost nothing, and medical
questions never reach the answer generator.
"""

import logging
from dataclasses import dataclass, field

from app.handoff import HandoffReason, asks_for_human, classify, handoff_message
from app.rag import answer_question

logger = logging.getLogger(__name__)


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

    # 2. Medical questions and indirect requests for a person.
    reason = classify(message)
    if reason is not None:
        return _handoff(reason, message)

    # 3. Normal question: answer from clinic data, or hand off if nothing relevant.
    rag = answer_question(message)
    if rag.answer is None:
        return _handoff(HandoffReason.NO_RELEVANT_INFO, message)

    return AssistantReply(answer=rag.answer, sources=rag.sources)
