"""Runtime configuration for the Document Q&A agent template."""

from dataclasses import dataclass

from framework.config import RuntimeConfig

default_config = RuntimeConfig()


@dataclass
class AgentMetadata:
    name: str = "Document Q&A Agent"
    version: str = "1.0.0"
    description: str = (
        "Retrieve, chunk, embed, and answer questions over documents using the Hive tool stack."
    )
    intro_message: str = (
        "Hi! I can help you answer questions over a document or a set of documents. "
        "Send me a PDF path, a URL, or raw text plus your question, and I’ll build the retrieval context."
    )


metadata = AgentMetadata()