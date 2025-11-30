# Project description

This MCP server provides a multi-agent, tool-augmented reasoning system designed for clinical research, biomedical analysis, and scientific information retrieval. It integrates structured clinical trial querying, PubMed literature search, PDF parsing, RAG-based document retrieval, and web research into a single orchestrated environment.

At its core is a Manager Agent (smolagents) that interprets user questions, plans multi-step reasoning, and coordinates two specialized agents:

1. Clinical Agent

Queries ClinicalTrials.gov with structured filters\
Retrieves PubMed articles\
Extracts text from scientific PDFs\
Produces structured tables, trial summaries, and evidence-based reports

2. Online Information Agent

Performs Wikipedia and DuckDuckGo searches\
Visits and extracts content from arbitrary webpages\
Supports general research, context building, and cross-verification

### The system includes a Gradio interface 
A real-time streaming of LLM reasoning, tools for creating/querying FAISS RAG vector stores, clinical trial search utilities, and detailed figure/image description.\
Instrumentation through OpenTelemetry and Langfuse provides end-to-end observability, ensuring transparent execution, step-level reasoning logs, and high-quality scientific outputs.\
In total, this MCP server delivers an extensible platform for evidence-based biomedical question answering, clinical trial intelligence, and automated scientific data extraction.

# Set-up in huggingchat
1. Use the specialized code agent from gradio interface [track agent reasonning]
* Must Add NEBIUS_API_KEY secrets to the huggingface space in order to use the Agent
2. Add MCP server to huggingchat [Choose an LLM to interact with the MCP tools]

# MCP Server — Detailed Technical Description

This MCP (Model Context Protocol) server provides a multi-agent orchestration framework optimized for clinical research, biomedical data extraction, scientific literature retrieval, and web-augmented analytical reasoning.
It integrates smolagents, LiteLLM, Langfuse, and OpenTelemetry to deliver a transparent, traceable, and high-accuracy computational research assistant.

The server exposes a Gradio-based user interface and multiple MCP API endpoints that allow other clients or LLM systems to interact with the agents programmatically.

## Core Capabilities

End-to-end clinical trial analysis with structured outputs\
Full-pipeline PubMed scientific literature retrieval\
Document parsing and PDF text extraction\
On-demand RAG (FAISS) vector store creation and querying\
Web search and webpage extraction for contextual enrichment\
Centralized multi-agent management and arbitration\
Real-time streaming of agent reasoning, tool calls, and final answers\
Full observability via OpenTelemetry instrumentation\
Trace and token logging via Langfuse

## Agent Architecture

The server implements a hierarchical agent system with a top-level Manager Agent coordinating specialized domain agents. Each agent operates under smolagents’ sandboxed code-execution environment with structured planning, controlled tool invocation, and deterministic step limits.

1. Manager Agent (High-Level Orchestrator)

Purpose:\
Acts as the decision-maker that interprets the user's question, plans a multi-step workflow, delegates subtasks, and synthesizes the final answer.\
Responsibilities:\
Perform question decomposition and reasoning planning\
Route relevant portions of the question to specialized agents\
Validate intermediate outputs for completeness and consistency\
Resolve conflicting information between agents\
Generate a consolidated final answer via FinalAnswerTool\

Configuration:\
max_steps=6\
Structured planning through planning_interval=3\
Verbose execution for full introspection (verbosity_level=0)\
Uses the high-capacity model openai/Qwen/Qwen3-Coder-480B-A35B-Instruct via Nebius\

2. Clinical Agent (Biomedical & Trial Retrieval Expert)

Purpose:\
Provides deep analysis of clinical trial datasets and scientific literature relevant to diseases, sponsors, phases, and study timelines.

Tools Available:\
ClinicalTrialsSearchTool — Queries the ClinicalTrials.gov API v2, returns structured TOON-formatted study objects\
search_pubmed — PubMed search by topic, author, or combined filters\
parse_pdf — Per-page text extraction for full-text clinical publications\

Typical Use Cases:\
Identifying phase-specific or sponsor-specific clinical trials\
Extracting structured study outcomes and metadata\
Connecting clinical trials to related scientific publications\
Summarizing disease-specific research landscapes\
Supporting systematic review tasks\

Execution Characteristics:\
Multi-step reasoning (up to 6 steps)\
Structured outputs to maintain data integrity\
Supports Python imports (NumPy, Pandas) for tabular processing

3. Online Information Agent (General Web Research Specialist)

### Purpose:
Augments biomedical reasoning with broader, up-to-date online information.

Tools Available:\
WikipediaSearchTool — High-level contextual information\
DuckDuckGoSearchTool — Recent news, recent studies, and web results\
VisitWebpageTool — Retrieve, extract, and summarize arbitrary URL content\
search_pubmed — Access to scientific publications\
parse_pdf — PDF ingestion for non-clinical research material\

### Typical Use Cases:

Gathering recent updates on treatment modalities\
Summarizing disease overviews or emerging therapies\
Extracting content from guidelines, web documents, or PDFs\
Cross-checking clinical trial information with external sources\

## Integrated Tools Exposed via MCP
### Clinical & Scientific Tools

ClinicalTrialsSearchTool (Primary clinical data feed)\
search_pubmed (PubMed literature)\
parse_pdf (Full-text document processing)\

### RAG / Vector Store Tools

create_rag: Builds FAISS vector store from DOIs/PMIDs/arXiv IDs\
use_rag: Retrieves top-k semantically relevant documents\

### Image & Figure Understanding

describe_figure: Generates detailed descriptions of scientific figures

### Web Research Tools

Wikipedia search\
DuckDuckGo search\
Webpage visiting & extraction\

## Gradio Application

The server exposes a multi-tab Gradio interface for interactive operations:

1. Question Analyzer

Streams LLM reasoning steps, tool invocations, and intermediate thoughts\
Displays final, aggregated answer from the manager agent\

2. Create RAG Vector Store

Upload list of DOIs/PMIDs\
Outputs path to stored FAISS index\

3. Query RAG Vector Store

Retrieve contextual scientific documents for any question

4. Clinical Trial Query Interface

Calls the ClinicalTrials.gov API\
Returns TOON-formatted study objects\

5. Figure Description

Accepts image uploads\
Produces detailed biomedical-quality captions\

## Summary

This MCP server is a full-stack multi-agent research system with:\
Hierarchical LLM planning\
Dedicated scientific and clinical tools\
Real-time execution monitoring\
FAISS-based custom RAG infrastructure\
Integrated web search and document extraction\
A complete interactive UI for researchers or clinicians\

It is suitable for:\
Clinical evidence synthesis\
Scientific research workflows\
Medical question answering\
Literature reviews\
Automated extraction pipelines\


## Git branches

main : main branch to merge development\
dev : auxiliary branches to add components\
production : branch to push on huggingface space [specific remote branch]\
\
When pushing from local production branch to huggingface space, need to specify to puch to main for automatic rebuilt/deployment\
\
Changes for productio includes:
1. app.py
- Guard function to insure clinical trials topic
- PATCH OpenInference 
- disbale tqmd
2. agent
- Change verbosity to 0 or 1
3. tool_create_FAISS_vector
- use smaller embedding model (cached inside the container) 
- autodetect device with get_device


