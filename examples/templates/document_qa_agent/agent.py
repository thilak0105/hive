"""Agent graph construction for the Document Q&A template."""

from __future__ import annotations

from framework.host.event_bus import EventBus
from framework.llm import LiteLLMProvider
from framework.loader.tool_registry import ToolRegistry
from framework.orchestrator import Constraint, EdgeCondition, EdgeSpec, Goal, NodeSpec, SuccessCriterion
from framework.orchestrator.edge import GraphSpec
from framework.orchestrator.orchestrator import ExecutionResult, Orchestrator
from framework.tracker.decision_tracker import DecisionTracker as Runtime

from .config import default_config, metadata

goal = Goal(
    id="document-qa",
    name="Document Q&A Agent",
    description=(
        "Answer questions over user-provided documents by extracting content, chunking it, "
        "embedding the chunks, retrieving relevant context, and generating a cited answer."
    ),
    success_criteria=[
        SuccessCriterion(
            id="sc-intake",
            description="Collects document source(s) and the user question",
            metric="source_capture",
            target="Yes",
            weight=0.2,
        ),
        SuccessCriterion(
            id="sc-chunking",
            description="Chunks extracted content into retrieval-friendly pieces",
            metric="chunking_done",
            target="Yes",
            weight=0.25,
        ),
        SuccessCriterion(
            id="sc-retrieval",
            description="Retrieves the most relevant chunks for the query",
            metric="retrieval_done",
            target="Yes",
            weight=0.25,
        ),
        SuccessCriterion(
            id="sc-answer",
            description="Produces a factual answer with cited evidence",
            metric="answer_cited",
            target="Yes",
            weight=0.3,
        ),
    ],
    constraints=[
        Constraint(
            id="c-no-fabrication",
            description="Never fabricate document content or citations",
            constraint_type="hard",
            category="quality",
        ),
        Constraint(
            id="c-source-grounding",
            description="Only answer from the provided document context",
            constraint_type="hard",
            category="quality",
        ),
    ],
)

intake_node = NodeSpec(
    id="intake",
    name="Intake",
    description="Collect the document source and the user question.",
    node_type="event_loop",
    client_facing=True,
    input_keys=[],
    output_keys=["doc_request"],
    system_prompt=(
        "You are the intake assistant for a Document Q&A agent.\n\n"
        "Ask the user for one of the following:\n"
        "- a PDF file path\n"
        "- a URL to scrape\n"
        "- raw text to analyze\n\n"
        "Also ask the user for the exact question they want answered.\n"
        "Once the user responds, call set_output(\"doc_request\", <JSON string>) with keys:\n"
        "- source_type: pdf | url | text\n"
        "- source: the file path / URL / text input\n"
        "- question: the user question\n"
        "- namespace: a short Pinecone namespace to isolate the retrieval index\n\n"
        "Keep the response brief and conversational."
    ),
    tools=[],
)

ingest_node = NodeSpec(
    id="ingest",
    name="Ingest",
    description="Extract text, chunk it, and embed the chunks for retrieval.",
    node_type="event_loop",
    input_keys=["doc_request"],
    output_keys=["indexed_context"],
    system_prompt=(
        "You are the ingestion node for a Document Q&A agent.\n\n"
        "Your job is to turn the user's source into retrieval-ready chunks.\n\n"
        "Use the right extraction tool based on the source_type:\n"
        "- pdf -> pdf_read\n"
        "- url -> web_scrape\n"
        "- text -> use the raw text directly\n\n"
        "Then use text_chunk_text to chunk the extracted text.\n"
        "For each chunk, call huggingface_run_embedding to create embeddings.\n"
        "Store the vectors in Pinecone using pinecone_create_index if needed, then pinecone_upsert_vectors.\n\n"
        "When done, call set_output(\"indexed_context\", <JSON string>) with:\n"
        "- namespace\n"
        "- question\n"
        "- chunk_count\n"
        "- a concise description of what was indexed\n\n"
        "Never fabricate chunk content. Keep the indexed_context compact."
    ),
    tools=[
        "pdf_read",
        "web_scrape",
        "text_chunk_text",
        "huggingface_run_embedding",
        "pinecone_create_index",
        "pinecone_upsert_vectors",
    ],
    client_facing=False,
)

retrieve_node = NodeSpec(
    id="retrieve",
    name="Retrieve",
    description="Retrieve the most relevant document chunks for the question.",
    node_type="event_loop",
    input_keys=["indexed_context"],
    output_keys=["retrieved_context"],
    system_prompt=(
        "You are the retrieval node for a Document Q&A agent.\n\n"
        "Embed the user's question with huggingface_run_embedding, then query Pinecone using pinecone_query_vectors.\n"
        "Return the top relevant chunks and any metadata that helps explain the answer.\n\n"
        "Call set_output(\"retrieved_context\", <JSON string>) with:\n"
        "- question\n"
        "- namespace\n"
        "- top_k\n"
        "- retrieved_chunks: a compact list of chunk records\n\n"
        "Only include chunks that were actually retrieved."
    ),
    tools=["huggingface_run_embedding", "pinecone_query_vectors"],
    client_facing=False,
)

synthesize_node = NodeSpec(
    id="synthesize",
    name="Synthesize",
    description="Answer the question using only the retrieved document context.",
    node_type="event_loop",
    input_keys=["retrieved_context"],
    output_keys=["answer"],
    system_prompt=(
        "You are the synthesis node for a Document Q&A agent.\n\n"
        "Use only the retrieved_context to answer the user's question.\n"
        "Cite the supporting chunk IDs or source references in the answer.\n"
        "If the retrieved context is insufficient, say so clearly instead of guessing.\n\n"
        "After writing the answer, call set_output(\"answer\", <JSON string>) with:\n"
        "- answer_text\n"
        "- citations\n"
        "- confidence\n\n"
        "Keep the final answer concise and grounded."
    ),
    tools=[],
    client_facing=False,
)

nodes = [intake_node, ingest_node, retrieve_node, synthesize_node]

edges = [
    EdgeSpec(
        id="intake-to-ingest",
        source="intake",
        target="ingest",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
    EdgeSpec(
        id="ingest-to-retrieve",
        source="ingest",
        target="retrieve",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
    EdgeSpec(
        id="retrieve-to-synthesize",
        source="retrieve",
        target="synthesize",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
]

entry_node = "intake"
entry_points = {"start": "intake"}
pause_nodes: list[str] = []
terminal_nodes = ["synthesize"]


class DocumentQAAgent:
    """Document Q&A agent template."""

    def __init__(self, config=None):
        self.config = config or default_config
        self.goal = goal
        self.nodes = nodes
        self.edges = edges
        self.entry_node = entry_node
        self.entry_points = entry_points
        self.pause_nodes = pause_nodes
        self.terminal_nodes = terminal_nodes
        self._executor: Orchestrator | None = None
        self._graph: GraphSpec | None = None
        self._event_bus: EventBus | None = None
        self._tool_registry: ToolRegistry | None = None

    def _build_graph(self) -> GraphSpec:
        return GraphSpec(
            id="document-qa-graph",
            goal_id=self.goal.id,
            version="1.0.0",
            entry_node=self.entry_node,
            entry_points=self.entry_points,
            terminal_nodes=self.terminal_nodes,
            pause_nodes=self.pause_nodes,
            nodes=self.nodes,
            edges=self.edges,
            default_model=self.config.model,
            max_tokens=self.config.max_tokens,
            loop_config={
                "max_iterations": 50,
                "max_tool_calls_per_turn": 30,
                "max_history_tokens": 32000,
            },
        )

    def _setup(self) -> Orchestrator:
        from pathlib import Path

        storage_path = Path.home() / ".hive" / "document_qa_agent"
        storage_path.mkdir(parents=True, exist_ok=True)

        self._event_bus = EventBus()
        self._tool_registry = ToolRegistry()

        mcp_config_path = Path(__file__).parent / "mcp_servers.json"
        if mcp_config_path.exists():
            self._tool_registry.load_mcp_config(mcp_config_path)

        llm = LiteLLMProvider(
            model=self.config.model,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
        )

        tool_executor = self._tool_registry.get_executor()
        tools = list(self._tool_registry.get_tools().values())

        self._graph = self._build_graph()
        runtime = Runtime(storage_path)

        self._executor = Orchestrator(
            runtime=runtime,
            llm=llm,
            tools=tools,
            tool_executor=tool_executor,
            event_bus=self._event_bus,
            storage_path=storage_path,
            loop_config=self._graph.loop_config,
        )

        return self._executor

    async def start(self) -> None:
        if self._executor is None:
            self._setup()

    async def stop(self) -> None:
        self._executor = None
        self._event_bus = None

    async def trigger_and_wait(
        self,
        entry_point: str,
        input_data: dict,
        timeout: float | None = None,
        session_state: dict | None = None,
    ) -> ExecutionResult | None:
        if self._executor is None:
            raise RuntimeError("Agent not started. Call start() first.")
        if self._graph is None:
            raise RuntimeError("Graph not built. Call start() first.")

        return await self._executor.execute(
            graph=self._graph,
            goal=self.goal,
            input_data=input_data,
            session_state=session_state,
        )

    async def run(self, context: dict, session_state=None) -> ExecutionResult:
        await self.start()
        try:
            result = await self.trigger_and_wait("start", context, session_state=session_state)
            return result or ExecutionResult(success=False, error="Execution timeout")
        finally:
            await self.stop()

    def info(self):
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "goal": {
                "name": self.goal.name,
                "description": self.goal.description,
            },
            "nodes": [n.id for n in self.nodes],
            "edges": [e.id for e in self.edges],
            "entry_node": self.entry_node,
            "entry_points": self.entry_points,
            "pause_nodes": self.pause_nodes,
            "terminal_nodes": self.terminal_nodes,
            "client_facing_nodes": [n.id for n in self.nodes if n.client_facing],
        }

    def validate(self):
        errors = []
        warnings = []

        if not self.nodes:
            errors.append("No nodes defined")
        if not self.edges:
            warnings.append("No edges defined")

        node_ids = {node.id for node in self.nodes}
        if self.entry_node not in node_ids:
            errors.append(f"Entry node '{self.entry_node}' not in nodes")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


default_agent = DocumentQAAgent()