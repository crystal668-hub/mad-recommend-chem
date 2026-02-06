# MAD (Multi-Agent Debate) for Electrocatalysis Literature

MAD is a multi-agent debate system for electrocatalysis analysis. Given **exactly 5 metal elements** (catalyst composition; optionally with relative percentages) and a **target reaction type**, the system debates and predicts the required performance metric(s) for that reaction (reaction-type specific; CO2RR includes main product + Faradaic efficiency).

It combines:
- 4 configurable LLM agents (LangChain tool-calling ReAct runtime)
- Chroma-backed RAG over a local Markdown literature corpus
- A debate coordinator (default: "LangGraph-style", implemented in-repo; no external `langgraph` dependency)
- Optional experience-store retrieval

## Quickstart

### 1) Install
Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

### 2) Configure API keys
Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
QWEN_API_KEY=...
VOYAGE_API_KEY=...
```

Notes:
- Agent 1/2/3 default to OpenRouter endpoints (`base_url: https://openrouter.ai/api/v1`).
- Agent 4 uses DashScope compatible-mode (`base_url: https://dashscope.aliyuncs.com/compatible-mode/v1`).
- See `config/config.yaml` for the exact mapping.

### 3) Prepare literature data
Place Markdown papers under:

```text
data/raw/CO2RR/*.md
data/raw/EOR/*.md
data/raw/HER/*.md
data/raw/HOR/*.md
data/raw/HZOR/*.md
data/raw/O5H/*.md
data/raw/OER/*.md
data/raw/ORR/*.md
data/raw/UOR/*.md
```

### 4) Build vector databases (Chroma)
Preferred: build per-agent collections via the batch script.

```bash
python build_vector_db_batch.py --agents agent1,agent2,agent3,agent4 --clear
```

Collections are stored in `vector_store.persist_directory` (default `./data/chroma_db`) and named:
`<vector_store.collection_name>_<agent_name>` (e.g., `electrochemistry_literature_agent1`).

Useful options:
- `--max-workers 1` (sequential build)
- `--embedding-batch-size 10`
- `--sleep-between-batches 0.5`

Legacy (single-agent) script:

```bash
python build_vector_db.py
```

### 5) Run a debate
Provide **exactly 5** metal elements (symbols only):

```bash
python main.py --components "Pt,Pd,Ru,Ir,Rh" --reaction-type CO2RR --engine langgraph
```

You may also provide relative percentages (the system will treat them as the electrode composition):

```bash
python main.py --components "Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)" --reaction-type OER --engine langgraph
```

Arguments:
- `--components`: comma-separated 5 metal elements
- `--reaction-type`: one of `CO2RR/EOR/HER/HOR/HZOR/O5H/OER/ORR/UOR` (recommended)
- `--engine`: `langgraph` (default) or `autogen` (optional; requires extra dependency installation)

### Outputs
- Results: `paths.outputs` (default `./outputs`) as `result_<timestamp>.json` (timestamp format: `YYYYMMDD_HHMMSS`)
- Logs:
  - rolling: `./logs/system.log`
  - per-run: `./logs/runs/<run_id>/run.log` (plus `events.jsonl`, `db.log`, `debate.log`)

## Configuration
All runtime configuration lives in `config/config.yaml`:
- `llm.*`: per-agent provider/model + embedding settings
- `vector_store.*`: Chroma persistence + base collection name
- `rag.*`: chunking + retrieval parameters
- `debate.*`: debate protocol parameters
- `paths.outputs`: output directory for saved results

## How it works (high level)
- `database/text_processor.py`: load + chunk Markdown documents (LlamaIndex parsers)
- `database/embedder.py`: multi-provider embeddings selected per agent
- `database/vector_store.py`: Chroma persistence with stable chunk ids
- `database/rag_system.py`: query embedding + Chroma similarity search
- `agents/react_agent.py`: LangChain tool-calling ReAct agent (`search_literature`, `search_experience`)
- `debate/langgraph_coordinator.py`: default debate coordinator and evidence enforcement

## Tests
```bash
python -m unittest discover -s test -p "test_*.py"
```

## License
MIT
