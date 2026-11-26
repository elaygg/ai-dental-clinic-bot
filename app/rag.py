"""Retrieval-augmented generation: find relevant clinic text, then ask the LLM.

Called by app/assistant.py after the handoff checks have passed.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.build_index import MANIFEST_FILE
from app.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a polite and professional administrator of Apostol Dental Clinic.
Answer the patient's question using ONLY the clinic information below.

Rules:
1. Be brief, friendly and to the point.
2. If the information below does not contain the answer, say so honestly.
   Never invent prices, doctors or services.
3. If prices are mentioned, state them clearly with the currency.
4. Never mention "context", "documents" or "knowledge base".
5. Reply in the same language as the patient's question.
6. End with one short question that helps the patient book a visit.

Clinic information:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


class IndexNotReadyError(RuntimeError):
    """The vector index is missing or was built with another embedding model."""


@dataclass
class RagAnswer:
    answer: str | None  # None = nothing relevant found, a person should answer
    sources: list[str]
    top_score: float | None


@lru_cache
def get_vector_store() -> Chroma:
    """Open the Chroma index once and reuse it for all requests."""
    settings = get_settings()
    manifest_path = settings.chroma_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise IndexNotReadyError("Vector index not found. Run: python -m app.build_index")

    built_with = json.loads(manifest_path.read_text())["embedding_model"]
    if built_with != settings.embedding_model:
        raise IndexNotReadyError(
            f"Index was built with '{built_with}', but EMBEDDING_MODEL is "
            f"'{settings.embedding_model}'. Rebuild it: python -m app.build_index"
        )

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        ),
        persist_directory=str(settings.chroma_dir),
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.llm_model, temperature=0.2, api_key=settings.openai_api_key)


def answer_question(question: str) -> RagAnswer:
    settings = get_settings()

    # 1. Retrieval: the top_k chunks closest in meaning to the question.
    results = get_vector_store().similarity_search_with_relevance_scores(question, k=settings.top_k)
    top_score = max((score for _, score in results), default=None)

    # 2. Drop chunks that are too far from the question.
    relevant = [doc for doc, score in results if score >= settings.relevance_threshold]

    # Log numbers only, never the question text: it may contain health data.
    logger.info(
        "question_chars=%d top_score=%s chunks_used=%d",
        len(question),
        f"{top_score:.3f}" if top_score is not None else None,
        len(relevant),
    )

    if not relevant:
        return RagAnswer(answer=None, sources=[], top_score=top_score)

    # 3. Generation: the LLM answers using only the retrieved chunks.
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant})
    return RagAnswer(answer=answer, sources=sources, top_score=top_score)
