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

from app.config import Settings, get_settings
from app.rag import IndexNotReadyError, answer_question

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dental Clinic AI Assistant",
    description="Answers patient questions about prices, treatments and payments using RAG.",
    version="1.0.0",
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


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask(request: AskRequest) -> AskResponse:
    try:
        result = answer_question(request.question)
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

    return AskResponse(answer=result.answer, sources=result.sources)
