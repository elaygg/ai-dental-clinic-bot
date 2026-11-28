# AI Dental Clinic Assistant

![CI](https://github.com/elaygg/ai-dental-clinic-bot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)

A chat assistant API for a dental clinic. It answers patient questions about prices, services,
doctors and payment options using RAG over the clinic's own information, and hands the chat
over to a human administrator when it should not answer by itself: medical questions,
booking requests, explicit requests for a person, or questions the clinic data does not cover.

> **Personal project.** The knowledge base is built from the public website and price list of
> Apostol Dental Clinic (Almaty, Kazakhstan). I am not affiliated with the clinic, and the bot
> is not used by it.

## Features

- **Answers from clinic data only (RAG).** The clinic's Markdown file is split into chunks,
  embedded with OpenAI and stored in ChromaDB. For each question the top 3 chunks above a
  relevance threshold are passed to `gpt-4o-mini`, which is told to use only that text.
- **Human handoff with a reason.** The API returns `handoff: true` and one of
  `medical_question`, `human_requested`, `booking_requested`, `no_relevant_info`,
  so the messenger integration can route the chat to an administrator.
- **Two-layer intent detection.** Keyword rules (English and Russian) catch explicit requests
  for a person instantly and for free. Everything else goes through an LLM classifier with
  structured output (a Pydantic schema, not free text).
- **Medical questions never reach the answer generator.** They are handed off before RAG runs,
  so the bot cannot give medical advice.
- **English and Russian.** Fixed handoff and greeting messages are chosen by the patient's language.
- **Production basics.** API key auth, request validation, clear error codes (401, 422, 502, 503),
  a `/health` endpoint, and logs that never contain the patient's message.
- **Docker.** The API runs in a container with Docker Compose: non-root user, health check,
  vector index on a volume, secrets passed at run time.
- **Tested.** 32 pytest tests run without an OpenAI key; GitHub Actions runs ruff, pytest
  and a Docker build on every pull request.

## How it works

```text
patient message (Telegram)
  │
  Make.com ── POST /chat ──► FastAPI
                               │
                               ├─ 1. keyword rules: "administrator", "call me back"...  → handoff: human_requested
                               │
                               ├─ 2. LLM intent classifier
                               │       medical_advice   → handoff: medical_question
                               │       human_request    → handoff: human_requested
                               │       booking_request  → handoff: booking_requested
                               │       small_talk       → fixed greeting
                               │       clinic_info      → step 3
                               │
                               └─ 3. RAG: ChromaDB top-3 chunks → gpt-4o-mini
                                       nothing relevant → handoff: no_relevant_info
                                       found            → answer + sources
```

The cheap checks run first, so explicit requests for a person cost no LLM calls.

## API

Interactive docs: `http://127.0.0.1:8000/docs` after starting the server.

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "How much is a kids dental cleaning?"}'
```

```json
{
  "answer": "A professional cleaning for children costs 17,000 ₸. Would you like to book a visit?",
  "sources": ["apostol_clinic_eng.md"],
  "handoff": false,
  "handoff_reason": null
}
```

A medical question is handed off instead of answered:

```json
{
  "answer": "This question needs a doctor's opinion, so I've passed it to our administrator. ...",
  "sources": [],
  "handoff": true,
  "handoff_reason": "medical_question"
}
```

| Status | When |
|---|---|
| `200` | Answer or handoff returned |
| `401` | `X-API-Key` header missing or wrong (only when `API_KEY` is set) |
| `422` | Empty question, or no `question` field |
| `502` | OpenAI request failed |
| `503` | Vector index not built (run `python -m app.build_index`) |

`POST /ask` is the same endpoint under its old name, kept for the existing Make.com scenario
and marked as deprecated. `GET /health` returns `{"status": "ok"}`.

## Getting started

Requires Python 3.13+ and an OpenAI API key.

```bash
git clone https://github.com/elaygg/ai-dental-clinic-bot.git
cd ai-dental-clinic-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then put your OPENAI_API_KEY in .env
python -m app.build_index     # builds the ChromaDB index from data/clinic/

python -m app.cli "How much is a kids dental cleaning?"   # ask from the terminal
uvicorn app.main:app --reload                             # or start the API
```

### Run with Docker

Requires Docker (Docker Desktop on macOS and Windows). No local Python needed.

```bash
cp .env.example .env                                     # then put your OPENAI_API_KEY in .env
docker compose build                                     # build the image
docker compose run --rm api python -m app.build_index    # build the index once, on the "index" volume
docker compose up -d                                     # start the API on http://127.0.0.1:8000

docker compose logs -f api                               # follow the logs
docker compose down                                      # stop (add -v to also delete the index)
```

Secrets are read from `.env` when the container starts and are never copied into the image
(see `.dockerignore`). The index lives on a named volume, so it survives restarts and image rebuilds.

### Configuration

All settings live in `app/config.py` and can be overridden in `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI access |
| `API_KEY` | empty (auth off) | Shared secret clients send in `X-API-Key` |
| `LLM_MODEL` | `gpt-4o-mini` | Answer generation and intent classification |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match the model the index was built with |
| `RELEVANCE_THRESHOLD` | `0.25` | Minimum cosine relevance for a chunk to be used |
| `CLINIC_PHONE` | clinic's public number | Shown to patients with urgent medical questions |
| `LOG_LEVEL` | `INFO` | Logging level |

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # 32 tests, about a second, no OpenAI key or internet needed
ruff check .      # lint
ruff format .     # format
```

The LLM classifier and the vector search are replaced with fakes (`monkeypatch`), so the tests
check the application's own logic:

- `tests/test_handoff.py`: keyword rules, language detection, handoff messages;
- `tests/test_assistant.py`: the decision flow for every intent, including that medical
  questions never reach RAG;
- `tests/test_api.py`: API key, validation, JSON fields, error codes, the deprecated `/ask` alias.

The quality of the LLM's answers and classifications is checked separately by asking real
questions through `app.cli`.

## Project structure

```text
app/
  config.py        settings from environment variables and .env
  build_index.py   splits the clinic Markdown into chunks and builds the ChromaDB index
  rag.py           retrieval + answer generation
  handoff.py       keyword rules, LLM intent classifier, handoff messages
  assistant.py     the decision flow: rules → classifier → RAG
  main.py          FastAPI app: /chat, /ask, /health
  cli.py           ask a question from the terminal
data/clinic/       clinic information in Markdown (public website data)
tests/             pytest suite
Dockerfile         the API image
compose.yaml       runs the image with .env and the index volume
.github/workflows/ CI: ruff, pytest, Docker build
```

## Messenger integration (Make.com)

The API only decides; the integration delivers. In Make.com:

1. **Telegram Bot: Watch Updates** receives the patient's message.
2. **HTTP: Make a request** sends `POST /chat` with `{"question": "{{message.text}}"}` and the
   `X-API-Key` header, with "Parse response" on.
3. **Router:**
   - `handoff = false`: send `answer` back to the patient;
   - `handoff = true`: send `answer` to the patient and notify the administrators' chat with
     `handoff_reason` and the question.

Because the API returns a flag instead of messaging anyone itself, another channel
(WhatsApp, Instagram) can be added in Make.com without changing the Python code.

## Design decisions

- **Rules first, LLM second.** Keyword rules are free and predictable but cannot understand
  meaning ("painless treatment?" is not a medical complaint), so they only catch explicit
  requests for a person. Meaning is left to the classifier.
- **Relevance threshold 0.25.** Measured on the clinic data: relevant questions score about
  0.64, unrelated ones about 0.14. Below the threshold the bot hands off instead of guessing.
- **Index manifest.** `build_index` records which embedding model built the index; the API
  refuses to start answering (503) if the configured model is different.
- **Index on a volume, not in the image.** Building the index needs the OpenAI key, and the key
  must never end up inside the image. So the image holds only code and clinic data, and the index
  is built once in the container and kept on a volume.
- **No patient text in logs.** Messages can contain health data, so logs keep only numbers
  (message length, relevance score, chunks used).
- **Stateless API.** The bot ends its answers with an offer to book, so a short "yes" is treated
  as a booking request. With conversation history this assumption would not be needed.

## Limitations and next steps

- Conversation history (for example in Redis), so follow-up questions keep their context.
- An evaluation set of labelled questions to measure classifier and answer quality.
- Real booking needs access to the clinic's schedule; for now requests go to an administrator.
- A fresh deployment. The first version ran on a DigitalOcean VPS (with ngrok for local
  testing); it is not deployed at the moment. With the Docker image, a new server needs only
  Docker, the repository and a `.env` file.

## Credits

The first RAG pipeline was based on
[pixegami/langchain-rag-tutorial](https://github.com/pixegami/langchain-rag-tutorial).
It has since been rewritten: settings, handoff, API, tests and CI are this project's own.
