"""FastAPI application: the HTTP API that Make.com (or any client) calls.

Run locally:
    uvicorn app.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from app.assistant import reply_to
from app.config import Settings, get_settings
from app.handoff import HandoffReason
from app.rag import IndexNotReadyError

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dental Clinic AI Assistant",
    description=(
        "Answers patient questions about prices, treatments and payments using RAG, "
        "and hands medical questions over to a human administrator."
    ),
    version="1.1.0",
)


class AskRequest(BaseModel):
    # Strip spaces first, so "   " counts as empty and is rejected.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000,
        examples=["How much does a kids' dental cleaning cost?"],
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    # true = a person must take over this chat (Make.com routes it to an administrator)
    handoff: bool
    handoff_reason: HandoffReason | None


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    provided: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Require the X-API-Key header when API_KEY is set in the environment."""
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    if not expected:
        return  # auth disabled, e.g. local development
    # compare_digest takes the same time for any input: no timing attacks.
    if not secrets.compare_digest((provided or "").encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check for monitoring and deploys."""
    return {"status": "ok"}


# One handler, two URLs. /chat is the main endpoint; /ask is kept so existing
# integrations (the Make.com scenario) keep working. Marked deprecated in /docs.
@app.post("/chat", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)], deprecated=True)
def ask(request: AskRequest) -> AskResponse:
    try:
        reply = reply_to(request.question)
    except IndexNotReadyError as exc:
        logger.error("%s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not ready",
        ) from exc
    except Exception as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service is temporarily unavailable",
        ) from exc

    return AskResponse(
        answer=reply.answer,
        sources=reply.sources,
        handoff=reply.handoff,
        handoff_reason=reply.handoff_reason,
    )
