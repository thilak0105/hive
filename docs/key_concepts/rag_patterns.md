# RAG Patterns

This document describes the initial Retrieval-Augmented Generation (RAG) pattern used by Hive templates.

## MVP Pattern

The first implementation follows a four-stage workflow:

1. **Intake**: Collect source type, source payload, and user question.
2. **Indexing**: Extract text, split into chunks, generate embeddings, and upsert vectors.
3. **Retrieval**: Embed the question and fetch top relevant chunks.
4. **Synthesis**: Generate an answer from retrieved context with citations.

## Why Chunking Matters

- Improves retrieval precision.
- Reduces embedding payload size.
- Enables transparent citations tied to chunk IDs.

## Baseline Strategies

- `sentence`: human-readable chunks, good default for prose.
- `paragraph`: larger context windows, useful for structured documents.
- `token`: deterministic sizing for prompt budget control.

## Grounding Rules

- Do not fabricate facts not present in retrieved chunks.
- Include citations in every substantive answer.
- If context is insufficient, explicitly communicate uncertainty.

## Future Extensions

- Hybrid retrieval (keyword + vector)
- Metadata filtering and reranking
- Retrieval quality metrics (e.g., recall@k, MRR)