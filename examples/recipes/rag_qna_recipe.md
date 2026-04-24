# RAG Q&A Recipe

This recipe walks through the MVP flow for the `document_qa_agent` template.

## Goal

Answer a user question using only content extracted from a provided document source.

## Inputs

- `source_type`: `pdf`, `url`, or `text`
- `source`: file path, URL, or raw text
- `question`: user question
- `namespace`: Pinecone namespace for indexed chunks

## Pipeline

1. Intake captures source + question.
2. Ingest extracts text (`pdf_read` or `web_scrape`), chunks it (`text_chunk_text`), creates embeddings (`huggingface_run_embedding`), then stores vectors (`pinecone_upsert_vectors`).
3. Retrieve embeds the question and fetches top matches (`pinecone_query_vectors`).
4. Synthesize answers with explicit citations to chunk IDs or source references.

## Required Credentials

- `PINECONE_API_KEY`
- `HUGGINGFACE_API_KEY`

## Example Test Prompt

"Use this document and answer: What are the top three recommendations? Return short bullet points and include citations."

## Success Checklist

- The answer is grounded in retrieved chunks.
- The response includes citations.
- The agent states uncertainty if retrieval context is insufficient.