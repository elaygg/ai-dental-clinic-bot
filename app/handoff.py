"""Human handoff: decide when the bot must stop and pass the chat to a person.

Two layers:
1. Keyword rules (free, instant, predictable) catch explicit requests
   for a human, in English and Russian.
2. An LLM classifier catches everything rules can't: medical questions
   ("my gum is swollen, what should I do?") and indirect requests for a person.

The API only returns a flag (handoff=true + reason). The actual routing to an
administrator happens in the messenger integration (a Make.com router).
"""

import re
from enum import StrEnum
from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings


class HandoffReason(StrEnum):
    HUMAN_REQUESTED = "human_requested"
    MEDICAL_QUESTION = "medical_question"
    NO_RELEVANT_INFO = "no_relevant_info"


# --- Layer 1: keyword rules -------------------------------------------------

HUMAN_REQUEST_PATTERNS = [
    # English
    r"\b(human|real person|live (agent|person)|operator|administrator|admin|manager)\b",
    r"\bcall me( back)?\b",
    # Russian
    r"администратор",
    r"оператор",
    r"менеджер",
    r"жив(ой|ого|ым) человек",
    r"перезвони",
    r"позвоните мне",
    r"свяжитесь со мной",
]
_HUMAN_REQUEST_RE = re.compile("|".join(HUMAN_REQUEST_PATTERNS), re.IGNORECASE)


def asks_for_human(text: str) -> bool:
    """True if the message explicitly asks for a person."""
    return _HUMAN_REQUEST_RE.search(text) is not None


# --- Layer 2: LLM classifier ------------------------------------------------

CLASSIFIER_PROMPT = """You route messages sent to a dental clinic's chat assistant.
Classify the patient's message into exactly one intent:

- clinic_info: general questions the clinic's information can answer: prices, services,
  doctors, opening hours, address, payment, booking, and how a treatment works in general,
  including whether it hurts, how long it takes, and anesthesia or sedation options.
- medical_advice: the patient describes a current health problem of their own or their
  child (pain, swelling, bleeding, fever, injury, a broken tooth) or asks for a diagnosis,
  what to do about it, or which medicine to take.
- human_request: the patient wants to talk to a person, an administrator or a doctor,
  or asks to be called back.

Examples:
"Is the treatment painless for kids?" -> clinic_info
"Do you treat teeth under sedation?" -> clinic_info
"How much is a zirconia crown?" -> clinic_info
"My tooth hurts when I drink something cold, what is it?" -> medical_advice
"My son has a fever after the extraction yesterday" -> medical_advice
"Can a doctor call me back?" -> human_request

If the patient describes a current symptom and you are unsure, choose medical_advice."""


class IntentResult(BaseModel):
    intent: Literal["clinic_info", "medical_advice", "human_request"]


@lru_cache
def get_classifier():
    """LLM that must answer with an IntentResult (structured output), not free text."""
    settings = get_settings()
    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    prompt = ChatPromptTemplate.from_messages([("system", CLASSIFIER_PROMPT), ("human", "{message}")])
    return prompt | llm.with_structured_output(IntentResult)


def classify(text: str) -> HandoffReason | None:
    """Ask the LLM whether the message needs a person. None = the bot can answer."""
    intent = get_classifier().invoke({"message": text}).intent
    if intent == "medical_advice":
        return HandoffReason.MEDICAL_QUESTION
    if intent == "human_request":
        return HandoffReason.HUMAN_REQUESTED
    return None


# --- What the patient sees --------------------------------------------------

_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

MESSAGES = {
    "en": {
        HandoffReason.HUMAN_REQUESTED: (
            "Of course! I've passed your request to our administrator, they will reply here shortly."
        ),
        HandoffReason.MEDICAL_QUESTION: (
            "This question needs a doctor's opinion, so I've passed it to our administrator. "
            "They will contact you shortly. If you have severe pain, swelling or bleeding, "
            "please call the clinic: {phone}."
        ),
        HandoffReason.NO_RELEVANT_INFO: (
            "I don't have this information, so I've passed your question to our administrator. "
            "They will reply here shortly."
        ),
    },
    "ru": {
        HandoffReason.HUMAN_REQUESTED: (
            "Конечно! Я передал ваш запрос администратору, он ответит вам здесь в ближайшее время."
        ),
        HandoffReason.MEDICAL_QUESTION: (
            "На этот вопрос должен ответить врач, поэтому я передал его администратору. "
            "С вами скоро свяжутся. Если у вас сильная боль, отёк или кровотечение, "
            "позвоните в клинику: {phone}."
        ),
        HandoffReason.NO_RELEVANT_INFO: (
            "У меня нет этой информации, поэтому я передал ваш вопрос администратору. "
            "Он ответит вам здесь в ближайшее время."
        ),
    },
}


def handoff_message(reason: HandoffReason, patient_text: str) -> str:
    """Reply in Russian if the patient wrote in Cyrillic, otherwise in English."""
    lang = "ru" if _CYRILLIC_RE.search(patient_text) else "en"
    return MESSAGES[lang][reason].format(phone=get_settings().clinic_phone)
