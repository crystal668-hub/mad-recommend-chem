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
- Agent 4 chats via OpenRouter (`model: qwen/qwen3-max-thinking`, `base_url: https://openrouter.ai/api/v1`).
- Agent 4 embeddings default to DashScope compatible-mode (`emb_url: https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings`), using `embedding_api_key` (typically `QWEN_API_KEY`).
- See `config/config.yaml` for the exact mapping.

### 3) Prepare literature data
Every supported Literature Type is declared once in
`database/literature_types.py`. Each entry points to a Markdown directory and a CSV
metadata file, for example:

```text
data/raw/OER/*.md                  metadata/OER.csv
data/raw/conductivity/*.md         metadata/Conductivity.csv
```

Every CSV must use this fixed schema:

```csv
file_name,doi,abstract
```

`file_name` is the local PDF name. The Markdown file must have the same basename and
the `.md` extension. DOI resolution prefers the CSV value, falls back to a DOI in the
Markdown body, and finally uses a stable `no-doi` identifier. `abstract` is validated
but is not copied into Chroma chunk metadata. The canonical Literature Type is stored
in the existing `reaction_type` metadata field for compatibility with existing
collections and retrieval filters.

### 4) Build vector databases (Chroma)
Preferred: build per-agent collections via the batch script.

```bash
python build_vector_db_batch.py --agents agent1,agent2,agent3,agent4 --clear
```

Collections are stored in `vector_store.persist_directory` (default `./data/chroma_db`) and named:
`<vector_store.collection_name>_<agent_name>` (e.g., `electrochemistry_literature_agent1`).

Useful options:
- `--embedding-request-batch-size 32` (override every provider profile)
- `--embedding-write-batch-size 100`
- `--embedding-global-max-inflight 8`

Offline embedding builds use native provider batches, shared endpoint/account quota
groups, centralized retries, strict vector validation, and a bounded single-writer
Chroma pipeline. Failed chunks are not stored as zero vectors; they remain missing for
`--resume` and are reported in `outputs/embedding_failures/*.jsonl`. The legacy
`--embedding-batch-size`, `--embedding-concurrency`, `--max-workers`, and
`--sleep-between-batches` controls remain accepted as deprecated compatibility options.
See `config/embedding_runtime.example.yaml` for explicit quota overrides.
The myrimate route for `google/gemini-embedding-2` is kept at request batch 1
because live validation returned only one vector for list input.

Legacy (single-agent) script:

```bash
python build_vector_db.py
```

### 5) Run a debate
Provide **exactly 5** metal elements (symbols only):

```bash
python main.py --components "Pt,Pd,Ru,Ir,Rh" --reaction-type CO2RR
```

You may also provide relative percentages (the system will treat them as the electrode composition):

```bash
python main.py --components "Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)" --reaction-type OER
```

Arguments:
- `--components`: comma-separated 5 metal elements
- `--reaction-type`: one of `CO2RR/EOR/HER/HOR/HZOR/O5H/OER/ORR/UOR` (recommended)
- `--engine`: `langgraph` (default; currently the only supported engine)

### 6) Rank reaction types (auto-run debates for each reaction)
If you want to **fix the composition** (5 metals + optional relative %) and let the system
run debates for **all reaction types** and return the **Top-K** reactions by grade:

```bash
python main.py --components "Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)" --rank-reactions
```

Optional controls:
- Subset of reactions:
  ```bash
  python main.py --components "Pt,Pd,Ru,Ir,Rh" --rank-reactions --reaction-types "OER,HER,ORR"
  ```
- Top-K (default 2):
  ```bash
  python main.py --components "Pt,Pd,Ru,Ir,Rh" --rank-reactions --top-k-reactions 3
  ```
- Reaction-level parallelism (default 1; higher may trigger API rate limits):
  ```bash
  python main.py --components "Pt,Pd,Ru,Ir,Rh" --rank-reactions --max-parallel-reactions 2
  ```
- Also save each per-reaction `outputs/result_*.json` (off by default):
  ```bash
  python main.py --components "Pt,Pd,Ru,Ir,Rh" --rank-reactions --save-each-reaction
  ```

Outputs:
- Ranking summary: `outputs/rank_<timestamp>.json`
- Logs and per-debate artifacts: under `logs/runs/<run_id>/`

### Outputs
- Results: `paths.outputs` (default `./outputs`) as `result_<timestamp>.json` (timestamp format: `YYYYMMDD_HHMMSS`)
- Logs:
  - rolling: `./logs/system.log`
  - per-run: `./logs/runs/<run_id>/run.log` (plus `events.jsonl`, `db.log`, `debate.log`)

## Configuration
All runtime configuration lives in `config/config.yaml`:
- `llm.*`: per-agent provider/model + embedding settings
- `embedding_runtime.*`: offline batch limits, quota groups, retries, and Chroma writer settings
- `vector_store.*`: Chroma persistence + base collection name
- `rag.*`: chunking + retrieval parameters
- `debate.*`: debate protocol parameters
- `paths.outputs`: output directory for saved results

## How it works (high level)
- `database/literature_types.py`: shared Literature Type directory + CSV configuration
- `database/text_processor.py`: load CSV-backed Markdown documents + chunk them (LlamaIndex parsers)
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
