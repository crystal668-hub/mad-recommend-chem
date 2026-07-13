# Unified Literature Type CSV Metadata Design

## Objective

Replace the separate reaction and category ingestion configurations with one shared
`LITERATURE_TYPE_CONFIGS` definition. Both vector database builders must load every
configured literature type through the same CSV-backed workflow while continuing to
write the established `reaction_type` field to Chroma metadata.

## Scope

This change covers:

- `build_vector_db_batch.py`, including removal of `--input-layout`.
- The legacy single-agent `build_vector_db.py`.
- The aggregate loading API in `database/text_processor.py`.
- Shared literature type configuration.
- Unit tests and builder documentation.

It does not rename the downstream Chroma `reaction_type` field or migrate existing
Chroma collections.

## Shared Configuration

Create `database/literature_types.py` as the single configuration source. Every entry
contains exactly the literature directory and its CSV metadata path. The obsolete
`"type": "fulltext"` property is removed.

```python
LITERATURE_TYPE_CONFIGS = {
    "CO2RR": {"path": "CO2RR", "metadata_csv": "./metadata/CO2RR.csv"},
    "EOR": {"path": "EOR", "metadata_csv": "./metadata/EOR.csv"},
    "HER": {"path": "HER", "metadata_csv": "./metadata/HER.csv"},
    "HOR": {"path": "HOR", "metadata_csv": "./metadata/HOR.csv"},
    "HZOR": {"path": "HZOR", "metadata_csv": "./metadata/HZOR.csv"},
    "O5H": {"path": "O5H", "metadata_csv": "./metadata/O5H.csv"},
    "OER": {"path": "OER", "metadata_csv": "./metadata/OER.csv"},
    "ORR": {"path": "ORR", "metadata_csv": "./metadata/ORR.csv"},
    "UOR": {"path": "UOR", "metadata_csv": "./metadata/UOR.csv"},
    "antiferromagnetism": {
        "path": "antiferromagnetism",
        "metadata_csv": "./metadata/Antiferromagnetism.csv",
    },
    "conductivity": {
        "path": "conductivity",
        "metadata_csv": "./metadata/Conductivity.csv",
    },
    "ferrimagnetism": {
        "path": "ferrimagnetism",
        "metadata_csv": "./metadata/Ferrimagnetism.csv",
    },
    "ferromagnetism": {
        "path": "ferromagnetism",
        "metadata_csv": "./metadata/Ferromagnetism.csv",
    },
    "photothermal conversion efficiency": {
        "path": "photothermal conversion efficiency",
        "metadata_csv": "./metadata/Photothermal conversion efficiency.csv",
    },
    "thermal conductivity": {
        "path": "thermal conductivity",
        "metadata_csv": "./metadata/Thermal Conductivity.csv",
    },
}
```

Configuration keys are the canonical values already consumed from
`metadata["reaction_type"]`. This preserves compatibility between old and newly built
collections.

## CSV Contract

Each configured CSV is UTF-8 or UTF-8 with BOM and has these required headers:

```csv
file_name,doi,abstract
```

- `file_name` is the local PDF file name, including `.pdf`.
- The corresponding Markdown file has the same basename and a `.md` extension.
- Matching removes each extension and compares normalized basenames
  case-insensitively.
- `doi` may be a bare DOI or a `doi.org` URL and is normalized with the existing DOI
  rules.
- `abstract` is read as part of the CSV record and its header is required, but its
  value may be empty.
- `abstract` is not copied into Document metadata or Chroma chunk metadata.

Duplicate normalized `file_name` basenames make the CSV ambiguous and therefore raise
an error. CSV rows without a corresponding Markdown file produce a warning and do not
create documents.

## Loading And Metadata Flow

`TextProcessor.load_literature_type_documents()` replaces the reaction/category
aggregate loaders. For each configured literature type it:

1. Resolves `data_dir/<path>` and the configured `metadata_csv`.
2. Validates that the CSV exists and contains all required headers.
3. Loads top-level `.md` files from the literature directory.
4. Matches each Markdown basename to the CSV `file_name` basename.
5. Uses the normalized CSV DOI when it is valid.
6. If the row is missing or its DOI is empty/invalid, extracts a DOI from the Markdown
   body; if none is found, creates the existing stable `no-doi` identifier.
7. Replaces source metadata with only `reaction_type` and `doc_id`.

The configured literature type directory may contain no Markdown files, in which case
the loader logs an error for that type and continues. A missing CSV, malformed header,
or duplicate normalized filename is a metadata contract violation and raises an error
instead of silently producing a partial type.

Chunking continues to copy `reaction_type` and `doc_id`, then adds the existing chunk
fields. The vector store receives the same downstream metadata schema as before:

```text
Document metadata -> chunk metadata -> VectorStore.add_documents -> Chroma
                         reaction_type remains unchanged
```

## Public API Changes

`build_vector_databases_batch()` removes these parameters without a compatibility
layer:

- `input_layout`
- `reaction_configs`
- `category_configs`

It adds `literature_type_configs`, defaulting to the shared configuration. The CLI
removes `--input-layout` entirely.

`build_vector_database()` removes `reaction_configs` and adds
`literature_type_configs`, also defaulting to the shared configuration.

Both builders call `load_literature_type_documents()` and report the expected directory
and CSV paths when the root data directory is unavailable.

The old aggregate methods `load_reaction_documents()` and
`load_category_documents()` are removed. The XLSX-specific loading and column-selection
code is removed because CSV is the only configured metadata format. The new ingestion
path does not use legacy TSV metadata lookup; Markdown DOI extraction and stable
`no-doi` generation remain available for fallback.

## Error Handling

- Missing root data directory: retain the builders' current early return behavior and
  log all configured directory/CSV expectations.
- Missing metadata CSV: raise `FileNotFoundError`.
- Missing required CSV header: raise `ValueError` naming the missing columns.
- Duplicate normalized `file_name`: raise `ValueError` naming the duplicate basename.
- Missing CSV row for a Markdown file: warn and use Markdown/no-DOI fallback.
- Empty or invalid CSV DOI: warn and use Markdown/no-DOI fallback.
- Extra CSV row without Markdown: warn and ignore it.

## Testing

Tests will establish these behaviors before implementation:

- The shared configuration contains all 15 existing literature types, every entry has
  `path` and `metadata_csv`, and no entry has `type` or `metadata_xlsx`.
- CSV PDF basenames match same-named Markdown files and produce normalized `doc_id`
  values.
- Existing canonical `reaction_type` values are preserved for both acronym and
  descriptive literature types.
- The required CSV headers, duplicate basenames, missing rows, invalid DOI fallback,
  and extra rows follow the defined policies.
- `abstract` is absent from Document and chunk metadata.
- Both vector builders call only the unified loader and accept only
  `literature_type_configs`.
- The batch CLI no longer advertises or accepts `--input-layout`.
- Focused tests and the full repository test suite pass before the implementation is
  committed.

## Documentation

README usage will remove `--input-layout` concepts and document the per-type directory
plus CSV requirement. Developer documentation will point ingestion maintenance to
`LITERATURE_TYPE_CONFIGS` and retain `reaction_type` as the retrieval compatibility
field.
