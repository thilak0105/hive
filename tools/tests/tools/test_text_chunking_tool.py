"""Tests for the text chunking tool."""

import pytest
from fastmcp import FastMCP

from aden_tools.tools.text_chunking_tool.text_chunking_tool import (
    chunker_split_documents,
    chunker_split_text,
    register_tools,
)


def test_sentence_strategy_splits_sentences() -> None:
    chunks = chunker_split_text("One. Two! Three?", strategy="sentence")

    assert chunks == ["One.", "Two!", "Three?"]


def test_paragraph_strategy_splits_paragraphs() -> None:
    chunks = chunker_split_text("First paragraph.\n\nSecond paragraph.", strategy="paragraph")

    assert chunks == ["First paragraph.", "Second paragraph."]


def test_token_strategy_respects_chunk_size_and_overlap() -> None:
    chunks = chunker_split_text(
        "one two three four five six",
        strategy="token",
        chunk_size=3,
        overlap=1,
    )

    assert chunks == ["one two three", "three four five", "five six"]


def test_document_chunking_preserves_metadata() -> None:
    chunked_documents = chunker_split_documents(
        [
            {
                "content": "Alpha. Beta.",
                "metadata": {"source": "sample.pdf"},
            }
        ],
        strategy="sentence",
    )

    assert chunked_documents == [
        {
            "content": "Alpha.",
            "chunk_id": "doc-0-chunk-0",
            "metadata": {
                "source": "sample.pdf",
                "document_index": 0,
                "chunk_index": 0,
                "strategy": "sentence",
            },
        },
        {
            "content": "Beta.",
            "chunk_id": "doc-0-chunk-1",
            "metadata": {
                "source": "sample.pdf",
                "document_index": 0,
                "chunk_index": 1,
                "strategy": "sentence",
            },
        },
    ]


def test_empty_text_returns_empty_chunks() -> None:
    assert chunker_split_text("   ") == []


def test_invalid_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="strategy must be one of"):
        chunker_split_text("hello world", strategy="invalid")


def test_register_tools_exposes_tool_functions(mcp: FastMCP) -> None:
    register_tools(mcp)

    assert "text_chunk_text" in mcp._tool_manager._tools
    assert "text_chunk_documents" in mcp._tool_manager._tools
