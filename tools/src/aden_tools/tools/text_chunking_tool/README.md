# Text Chunking Tool

This tool splits text or document payloads into smaller chunks for retrieval-augmented workflows.

## Functions

- `text_chunk_text`
- `text_chunk_documents`

## Strategies

- `sentence`: split on sentence boundaries
- `paragraph`: split on paragraph boundaries
- `token`: split by token count with optional overlap

## Notes

- Metadata is preserved for document chunks.
- Empty inputs return empty chunk lists.
- Use `token` strategy for the first MVP when you need deterministic chunk sizing.