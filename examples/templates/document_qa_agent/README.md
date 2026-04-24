# Document Q&A Agent

This template demonstrates a minimal end-to-end retrieval workflow for question answering over PDFs, URLs, or raw text.

## Included in PR 1

- Text chunking utility
- Agent template skeleton
- MCP configuration for existing Hive tools plus the new chunking tool

## Required tools

- `pdf_read`
- `web_scrape`
- `text_chunk_text`
- `huggingface_run_embedding`
- `pinecone_create_index`
- `pinecone_upsert_vectors`
- `pinecone_query_vectors`

## Run

```bash
cd examples/templates/document_qa_agent
uv run python -m document_qa_agent info
```

## Notes

- `INCLUDE_UNVERIFIED_TOOLS=true` is set in `mcp_servers.json` so the new chunking tool is exposed by the shared tools server.
- The first PR keeps ChromaDB and advanced RAG evaluation out of scope.