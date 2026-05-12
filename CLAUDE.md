# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic RAG system built with FastAPI and LangGraph. It provides a streaming chat API that answers questions using an uploaded document knowledge base, with human-in-the-loop clarification, query rewriting, and a two-level parent-child chunk retrieval strategy.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server (from project/ directory)
cd project && python app.py
# Or directly with uvicorn
cd project && uvicorn api_server:app --host 0.0.0.0 --port 8000
```

There is no test suite, lint config, or build script in this repository.

## Environment Variables

Copy `.env_example` to `.env` and set:

- `DASHSCOPE_BASE_URL` — DashScope compatible-mode endpoint
- `DASHSCOPE_API_KEY` — API key for LLM, embeddings, and reranker
- `LLM_MODEL` — Defaults to `qwen3-max`
- `LLM_TEMPERATURE` — Defaults to `0`
- `COMPANY_NAME` — Used in system prompts

## Architecture

### Directory Layout

All application code lives under `project/`:

- `api_server.py` — FastAPI app factory; wires CORS, routers, and lifespan
- `api/dependencies.py` — Lifespan initializes `RAGSystem`, `ChatInterface`, and `DocumentManager` into `app.state`
- `api/routes/chat.py` — `POST /chat` SSE streaming endpoint
- `api/routes/documents.py` — `GET /documents`, `POST /documents`, `DELETE /documents`
- `core/rag_system.py` — Central coordinator: owns vector DB, parent store, chunker, and the compiled LangGraph
- `core/chat_interface.py` — Wraps the graph with `async_stream_chat` (SSE) and `chat` (sync)
- `core/document_manager.py` — Handles PDF/DOCX/Markdown ingestion, conversion, chunking, and storage
- `db/vector_db_manager.py` — Qdrant client with custom `DashScopeEmbeddings`
- `db/parent_store_manager.py` — File-system JSON store for parent chunk documents
- `document_chunker.py` — Hierarchical parent-child chunking using `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
- `rag_agent/graph.py` — Compiles the two-level LangGraph (main graph + per-question agent subgraph)
- `rag_agent/nodes.py` — All graph nodes: summarize, rewrite, agent, retry, aggregate, etc.
- `rag_agent/edges.py` — Conditional edges: routes rewritten questions to parallel agent subgraphs
- `rag_agent/tools.py` — `ToolFactory` exposing `search_child_chunks` and `retrieve_parent_chunks`
- `rag_agent/prompts.py` — All Chinese-language system prompts
- `rag_agent/schemas.py` — Pydantic models for structured LLM outputs
- `util.py` — PDF conversion pipeline (simple/medium/complex) and DOCX conversion

### LangGraph Structure

The graph is two-level:

1. **Main graph** (`State`) — per conversation:
   - `summarize` → `analyze_rewrite` → [`human_input` | `process_question`]
   - `process_question` is a compiled subgraph (one instance per rewritten question, executed in parallel via `Send`)
   - `aggregate` synthesizes all subgraph answers into a final response
   - `interrupt_before=["human_input"]` enables human-in-the-loop when the query is unclear

2. **Agent subgraph** (`AgentState`) — per rewritten question:
   - `local_agent` → (`tools_condition`) → `local_tools` → `local_agent`
   - After tool loop ends: `check_local_result` evaluates whether the answer is sufficient
   - If insufficient and `retry_count < MAX_LOCAL_RETRIES`: routes to `prepare_retry` then back to `local_agent`
   - Otherwise: `extract_answer` emits the final answer for this question

### Document Ingestion Pipeline

1. PDFs are analyzed and routed to one of three converters:
   - **Simple** — `pymupdf4llm` (fast text extraction)
   - **Medium** — Docling (OCR + table structure, falls back to simple)
   - **Complex** — VLM (`qwen3-vl-30b-a3b-instruct`) page-by-page via DashScope (falls back to simple)
2. Markdown output is written to `markdown_docs/`
3. `DocumentChunker` splits by markdown headers into parent chunks, then recursively into child chunks
4. Child chunks are embedded (DashScope `text-embedding-v4`) and stored in Qdrant
5. Parent chunks are stored as JSON files in `parent_store/`

### Retrieval Strategy

- `search_child_chunks` — similarity search in Qdrant (broad `k*3` candidate fetch, then reranked via DashScope `gte-rerank`)
- `retrieve_parent_chunks` — fetches the full parent document by `parent_id` from the file-system store
- The agent may call both tools within its retry loop, varying keywords on retry

### Streaming Chat Protocol

`POST /chat` returns SSE. Events are JSON objects with `type`:

- `token` — streamed LLM output tokens from the `aggregate` node
- `node_end` — a graph node completed (`summarize`, `analyze_rewrite`, `process_question`, `aggregate`)
- `interrupt` — graph paused at `human_input`; client should send follow-up with the same `thread_id`
- `done` — conversation turn complete
- `error` — failure

If no `thread_id` is provided, a new thread is created. Pass the returned `thread_id` to continue an interrupted conversation.

### Key Configuration

All tunables are in `project/config.py`:

- `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` — child splitter settings
- `MIN_PARENT_SIZE` / `MAX_PARENT_SIZE` — parent chunk bounds
- `MAX_LOCAL_RETRIES` — agent subgraph retry limit
- `CHINESE_SEPARATORS` — recursive splitter separator list for Chinese text

### Mermaid Diagrams

On startup, `create_agent_graph` writes `agent_graph.mmd` and `agent_subgraph.mmd` (with a blue theme) to the repository root for visual inspection.
