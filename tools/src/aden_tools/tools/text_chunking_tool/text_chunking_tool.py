"""Text chunking utilities for retrieval-augmented workflows."""

from __future__ import annotations

import re

from fastmcp import FastMCP

_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"\n\s*\n+")


def _normalize_text(text: str) -> str:
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_BOUNDARY_PATTERN.split(text) if segment.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [segment.strip() for segment in _PARAGRAPH_BOUNDARY_PATTERN.split(text) if segment.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


def _split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []

    effective_overlap = min(max(overlap, 0), max(chunk_size - 1, 0))
    step = max(chunk_size - effective_overlap, 1)
    chunks: list[str] = []

    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break

    return chunks


def chunker_split_text(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into semantically meaningful chunks."""

    text = _normalize_text(text)
    if not text:
        return []

    if chunk_size < 1:
        return [text]

    if strategy == "sentence":
        return _split_sentences(text)
    if strategy == "paragraph":
        return _split_paragraphs(text)
    if strategy == "token":
        return _split_tokens(text, chunk_size=chunk_size, overlap=overlap)

    raise ValueError("strategy must be one of: sentence, paragraph, token")


def chunker_split_documents(
    documents: list[dict],
    strategy: str = "sentence",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Split documents into chunk dictionaries while preserving metadata."""

    chunked_documents: list[dict] = []

    for document_index, document in enumerate(documents):
        content = str(document.get("content", ""))
        metadata = dict(document.get("metadata", {}))
        metadata.setdefault("document_index", document_index)

        for chunk_index, chunk in enumerate(
            chunker_split_text(
                content,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        ):
            chunked_documents.append(
                {
                    "content": chunk,
                    "chunk_id": f"doc-{document_index}-chunk-{chunk_index}",
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "strategy": strategy,
                    },
                }
            )

    return chunked_documents


def register_tools(mcp: FastMCP) -> None:
    """Register chunking tools with the MCP server."""

    @mcp.tool()
    def text_chunk_text(
        text: str,
        strategy: str = "sentence",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[str]:
        """Split a single text input into chunks."""

        return chunker_split_text(
            text=text,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    @mcp.tool()
    def text_chunk_documents(
        documents: list[dict],
        strategy: str = "sentence",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[dict]:
        """Split a list of documents into chunk dictionaries."""

        return chunker_split_documents(
            documents=documents,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
