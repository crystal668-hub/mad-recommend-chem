# Paper Parse Contract

## Input

- `--input`: local file path
- `--output-dir`: directory for generated artifacts
- optional config fields can be supplied via `--config-json`

## Output JSON

- `document_id`
- `fulltext_status`
- `source_artifact_path`
- `fulltext_artifact_path`
- `sections_artifact_path`
- `snippets_artifact_path`
- `extraction_report_path`
- `sections`
- `warnings`
- `extractor`
- `ocr_applied`
- `report`

## Status Values

- `fulltext_indexed`
- `fulltext_unusable`
- `binary_only`

## Parser Policy

- Primary backend is always `pymupdf`
- Secondary backend defaults to `docling`
- If PyMuPDF is unusable or rejected by quality gates, Docling is attempted
- No repository-local imports or runtime state are required

