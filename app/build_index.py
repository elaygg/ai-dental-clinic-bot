"""Build the Chroma vector index from the Markdown files in data/clinic/.

Run it once, and again after every change to the clinic data:

    python -m app.build_index

The first version of this pipeline followed Pixegami's LangChain RAG tutorial:
https://github.com/pixegami/langchain-rag-tutorial
"""

import json
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import Settings, get_settings

# Written next to the index; the API reads it to check the embedding model.
MANIFEST_FILE = "index_manifest.json"

HEADERS_TO_SPLIT_ON = [("#", "section"), ("##", "subsection"), ("###", "topic")]


def load_documents(settings: Settings) -> list[Document]:
    """Read every .md file in the data folder as one Document."""
    files = sorted(settings.data_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files found in {settings.data_dir}")
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in files
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split by Markdown headers first, then cut long sections into smaller chunks.

    Each chunk starts with its header path ("SERVICES AND PRICES > Therapy"),
    so a price line keeps the context of the section it belongs to.
    """
    header_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

    chunks: list[Document] = []
    for doc in documents:
        for section in header_splitter.split_text(doc.page_content):
            headers = [section.metadata[key] for _, key in HEADERS_TO_SPLIT_ON if key in section.metadata]
            breadcrumb = " > ".join(headers)
            for piece in size_splitter.split_text(section.page_content):
                chunks.append(
                    Document(
                        page_content=f"{breadcrumb}\n{piece}" if breadcrumb else piece,
                        metadata={"source": doc.metadata["source"], "section": breadcrumb},
                    )
                )
    return chunks


def build_index(settings: Settings) -> int:
    documents = load_documents(settings)
    chunks = split_documents(documents)
    print(f"Split {len(documents)} file(s) into {len(chunks)} chunks.")

    # Start from a clean index so deleted or changed text does not stay behind.
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        ),
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        # Cosine distance, so relevance scores are in 0..1 and easy to read.
        collection_metadata={"hnsw:space": "cosine"},
    )

    manifest = {"embedding_model": settings.embedding_model, "chunks": len(chunks)}
    (settings.chroma_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(chunks)} chunks to {settings.chroma_dir}.")
    return len(chunks)


if __name__ == "__main__":
    build_index(get_settings())
