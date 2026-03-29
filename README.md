# ChemQA

ChemQA is a chemistry literature QA system built around the `react_reviewed` workflow.


## Install

Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

## Configure

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
QWEN_API_KEY=...
VOYAGE_API_KEY=...
OPENALEX_MAILTO=you@example.org
CROSSREF_MAILTO=you@example.org
SEMANTIC_SCHOLAR_API_KEY=...
UNPAYWALL_EMAIL=you@example.org
```

## Run ChemQA

```bash
python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --save-output
```

Optional flags:

- `--context "..."`
- `--artifact-dir <path>`
- `--config <path>`

Outputs:

- run artifacts: `logs/runs/<run_id>/qa_artifacts/`
- user-facing result: `outputs/qa_result_<timestamp>.json`

## Live Validation

Run the standard live-validation entrypoint:

```bash
python -m chemqa.live_validation --question "Does Pt/C improve HER activity in 1 M KOH?"
```

ChemBench reasoning smoke tests are available as a separate runner:

```bash
python -m qa.chembench_smoke
```

Optional flags:

- `--cases-file ./evals/chembench_smoke_cases.yaml`
- `--case analytical_chemistry_27`
- `--artifact-root ./outputs/chembench_smoke/manual_run`
- `--shadow-config <path>`

## Configuration

Runtime configuration lives in `config/config.yaml`.

Primary sections:

- `llm.*`: provider and model settings
- `qa.*`: `react_reviewed` runtime controls and provider settings
- `logging.*`: log destinations and format
- `paths.outputs`: exported result directory

Important QA provider controls:

- `qa.providers.http_timeout`
- `qa.providers.fetch_timeout`
- `qa.providers.document_fetch_timeout_seconds`
- `qa.providers.document_fetch_total_timeout_seconds`
- `qa.providers.provider_redirect_limit`
- `qa.providers.retry_attempts`

Important `react_reviewed` controls:

- `qa.react_reviewed.max_propose_steps_initial`
- `qa.react_reviewed.max_propose_steps_revision`
- `qa.react_reviewed.stage_watchdog_seconds`

## Logs

- rolling log: `logs/system.log`
- per-run text log: `logs/runs/<run_id>/run.log`
- per-run structured log: `logs/runs/<run_id>/events.jsonl`

## Tests

```bash
python -m unittest discover -s test -p "test_*.py"
```
