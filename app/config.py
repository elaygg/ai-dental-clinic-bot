"""Application settings, read from environment variables and the .env file.

Every value has one source of truth here, so the indexing script, the CLI
and the API can never disagree (for example, on the embedding model).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: app/config.py -> app/ -> repo root
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Secrets. SecretStr hides the value in logs and error messages.
    openai_api_key: SecretStr
    # Optional shared secret for our own API. Empty = no auth (local development).
    api_key: SecretStr | None = None

    # Models. The index must be queried with the same embedding model
    # it was built with; change it only together with a rebuild.
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Knowledge base and vector index
    data_dir: Path = BASE_DIR / "data" / "clinic"
    chroma_dir: Path = BASE_DIR / "chroma"
    collection_name: str = "clinic"

    # Retrieval
    top_k: int = 3
    # Minimum cosine relevance (0..1) for a chunk to be used as context.
    # Tune it by looking at the top_score values in the logs.
    relevance_threshold: float = 0.25

    # Shown to patients with urgent medical questions (public clinic number).
    clinic_phone: str = "+7 707 477 2888"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Read settings once and reuse them."""
    # pydantic-settings fills openai_api_key from the environment or .env at runtime.
    # The type checker cannot see that and reports a missing argument, so we silence it here.
    return Settings()  # pyright: ignore[reportCallIssue]
