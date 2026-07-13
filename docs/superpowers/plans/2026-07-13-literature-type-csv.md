# Unified Literature Type CSV Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reaction/category ingestion modes with one shared Literature Type configuration whose documents all use `file_name,doi,abstract` CSV metadata while preserving Chroma `reaction_type` metadata.

**Architecture:** `database/literature_types.py` owns the only ingestion configuration. `TextProcessor` loads each configured directory through one CSV-aware path, resolves DOI as CSV then Markdown then stable `no-doi`, and emits only `reaction_type` and `doc_id`; both builders call that same API.

**Tech Stack:** Python 3, standard-library `csv`, LlamaIndex `SimpleDirectoryReader`, `unittest`, ChromaDB.

---

## File Structure

- Create `database/literature_types.py`: canonical Literature Type configuration shared by all builders.
- Create `test/test_literature_type_configs.py`: configuration schema and coverage tests.
- Modify `database/text_processor.py`: CSV parsing and unified literature loading; remove XLSX loaders and old aggregate loaders.
- Modify `test/test_text_processor_doi_extraction.py`: replace XLSX/category tests with CSV Literature Type tests.
- Rename `test/test_build_vector_db_batch_input_layout.py` to `test/test_build_vector_db_batch_literature_types.py`: unified batch API and CLI tests.
- Modify `build_vector_db_batch.py`: remove layout selection and consume the shared configuration.
- Create `test/test_build_vector_db_literature_types.py`: legacy single-builder API and loader tests.
- Modify `build_vector_db.py`: consume the shared configuration.
- Modify `README.md`: document unified directory/CSV inputs.
- Modify `Developer.md`: update ingestion configuration maintenance guidance.

### Task 1: Add The Shared Literature Type Configuration

**Files:**
- Create: `database/literature_types.py`
- Create: `test/test_literature_type_configs.py`

- [ ] **Step 1: Write the failing configuration test**

```python
import unittest

from database.literature_types import LITERATURE_TYPE_CONFIGS


class LiteratureTypeConfigsTests(unittest.TestCase):
    def test_contains_all_existing_types_with_csv_metadata(self):
        expected = {
            "CO2RR", "EOR", "HER", "HOR", "HZOR", "O5H", "OER", "ORR", "UOR",
            "antiferromagnetism", "conductivity", "ferrimagnetism", "ferromagnetism",
            "photothermal conversion efficiency", "thermal conductivity",
        }
        self.assertEqual(set(LITERATURE_TYPE_CONFIGS), expected)
        for config in LITERATURE_TYPE_CONFIGS.values():
            self.assertEqual(set(config), {"path", "metadata_csv"})
            self.assertTrue(config["metadata_csv"].endswith(".csv"))
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest test.test_literature_type_configs -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'database.literature_types'`.

- [ ] **Step 3: Implement the shared configuration**

```python
from __future__ import annotations

from typing import Dict


LITERATURE_TYPE_CONFIGS: Dict[str, Dict[str, str]] = {
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

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest test.test_literature_type_configs -v`

Expected: PASS, 1 test.

Run: `python -m unittest discover -s test -p "test_*.py"`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```powershell
git add database/literature_types.py test/test_literature_type_configs.py
git commit -m "feat: add shared literature type configuration"
```

### Task 2: Implement CSV Metadata Loading

**Files:**
- Modify: `database/text_processor.py`
- Modify: `test/test_text_processor_doi_extraction.py`

- [ ] **Step 1: Replace XLSX tests with failing CSV mapping tests**

Remove the pandas/XLSX test setup and add a UTF-8 CSV helper using `csv.DictWriter`. Add tests that create `paper_001.md` and a CSV row for `paper_001.pdf`, then assert:

```python
docs = TextProcessor(data_dir=str(root / "data")).load_literature_type_directory_documents(
    data_dir=str(data_dir),
    metadata_csv_path=str(csv_path),
    literature_type="antiferromagnetism",
)
self.assertEqual(docs[0].metadata, {
    "reaction_type": "antiferromagnetism",
    "doc_id": "10.1002/cctc.202200897",
})
self.assertNotIn("abstract", docs[0].metadata)
```

Add separate tests for:

```python
# Missing CSV row falls back to Markdown DOI.
self.assertEqual(docs[0].metadata["doc_id"], "10.1021/acs.jacs.0c00000")

# Invalid CSV DOI and no Markdown DOI use a stable no-doi id.
self.assertTrue(docs[0].metadata["doc_id"].startswith("no-doi:paper_003_"))

# A CSV missing abstract is rejected.
with self.assertRaisesRegex(ValueError, "missing required columns: abstract"):
    processor.load_literature_type_directory_documents(
        data_dir=str(data_dir),
        metadata_csv_path=str(csv_path),
        literature_type="OER",
    )

# PDF names differing only by case are duplicate normalized basenames.
with self.assertRaisesRegex(ValueError, "Duplicate file_name basename"):
    processor.load_literature_type_directory_documents(
        data_dir=str(data_dir),
        metadata_csv_path=str(csv_path),
        literature_type="OER",
    )
```

- [ ] **Step 2: Run the CSV tests and verify RED**

Run: `python -m unittest test.test_text_processor_doi_extraction.TextProcessorLiteratureTypeDocumentTests -v`

Expected: FAIL because `load_literature_type_directory_documents` does not exist.

- [ ] **Step 3: Add the CSV parser**

Import `csv` and add a helper with this interface:

```python
def _load_metadata_csv_index(self, metadata_csv_path: str) -> Dict[str, Dict[str, str]]:
    csv_path = Path(metadata_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Metadata CSV path is not a file: {metadata_csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {str(name or "").strip() for name in (reader.fieldnames or [])}
        missing = sorted({"file_name", "doi", "abstract"} - headers)
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {', '.join(missing)}")

        index: Dict[str, Dict[str, str]] = {}
        for row in reader:
            file_name = str(row.get("file_name") or "").strip()
            if not file_name:
                continue
            key = Path(file_name).stem.casefold()
            if key in index:
                raise ValueError(f"Duplicate file_name basename in metadata CSV: {key}")
            index[key] = {
                "file_name": file_name,
                "doi": str(row.get("doi") or "").strip(),
                "abstract": str(row.get("abstract") or "").strip(),
            }
    return index
```

- [ ] **Step 4: Add the unified per-directory loader**

Implement `load_literature_type_directory_documents(data_dir, metadata_csv_path, literature_type)` using top-level `.md` files and `SimpleDirectoryReader`. Resolve each document as follows:

```python
row = metadata_index.get(file_stem.casefold())
doc_id = self._normalize_doi((row or {}).get("doi", ""))
if not doc_id:
    doc_id = self._extract_doi_from_markdown(doc.text)
if not doc_id:
    doc_id = self._build_fallback_doc_id(file_name, file_path_str)
doc.metadata = {
    "reaction_type": literature_type,
    "doc_id": doc_id,
}
```

Track matched CSV keys. Warn once per missing Markdown mapping or invalid CSV DOI, and warn for `set(metadata_index) - matched_keys`. Do not place the CSV row or `abstract` into `doc.metadata`.

- [ ] **Step 5: Add the aggregate loader and remove superseded XLSX APIs**

Add:

```python
def load_literature_type_documents(
    self,
    base_dir: str,
    literature_type_configs: Dict[str, Dict],
) -> List[Document]:
    all_documents: List[Document] = []
    base_path = Path(base_dir)
    for literature_type, config in literature_type_configs.items():
        docs = self.load_literature_type_directory_documents(
            data_dir=str(base_path / config.get("path", literature_type)),
            metadata_csv_path=str(config["metadata_csv"]),
            literature_type=literature_type,
        )
        all_documents.extend(docs)
    return all_documents
```

Remove `_metadata_xlsx_index`, `_load_metadata_xlsx_index`, `_select_xlsx_id_column`, `load_category_documents`, `load_category_directory_documents`, and `load_reaction_documents`. Keep Markdown DOI extraction and stable fallback helpers. The unified loader must call Markdown extraction directly and must not call the legacy TSV lookup.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m unittest test.test_text_processor_doi_extraction -v`

Expected: PASS for DOI extraction and all CSV Literature Type cases.

Run: `python -m unittest discover -s test -p "test_*.py"`

Expected: the old batch input-layout tests may fail because they still reference removed loaders; all other tests PASS. If unittest reports those expected failures, proceed immediately to Task 3 without committing Task 2 separately, because `AGENTS.md` requires tests to pass before any commit. Otherwise commit Task 2 now.

- [ ] **Step 7: Commit only when the full suite is green**

If Task 2 remains uncommitted due to expected batch test failures, include its files in Task 3's commit. If the full suite is green:

```powershell
git add database/text_processor.py test/test_text_processor_doi_extraction.py
git commit -m "feat: load literature metadata from csv"
```

### Task 3: Unify The Batch Builder

**Files:**
- Rename: `test/test_build_vector_db_batch_input_layout.py` to `test/test_build_vector_db_batch_literature_types.py`
- Modify: `test/test_build_vector_db_batch_literature_types.py`
- Modify: `build_vector_db_batch.py`

- [ ] **Step 1: Replace layout tests with failing unified API tests**

The builder test must assert one loader call:

```python
batch.build_vector_databases_batch(
    data_dir=str(data_dir),
    literature_type_configs=batch.LITERATURE_TYPE_CONFIGS,
    agent_names=["agent1"],
)
processor.load_literature_type_documents.assert_called_once_with(
    base_dir=str(data_dir),
    literature_type_configs=batch.LITERATURE_TYPE_CONFIGS,
)
```

Add signature assertions:

```python
parameters = inspect.signature(batch.build_vector_databases_batch).parameters
self.assertIn("literature_type_configs", parameters)
self.assertNotIn("input_layout", parameters)
self.assertNotIn("reaction_configs", parameters)
self.assertNotIn("category_configs", parameters)
```

Add a CLI rejection test by patching `sys.argv` to include `--input-layout reaction`, redirecting stderr, and asserting `batch.main()` raises `SystemExit` with code `2` and reports `unrecognized arguments`.

- [ ] **Step 2: Run batch tests and verify RED**

Run: `python -m unittest test.test_build_vector_db_batch_literature_types -v`

Expected: FAIL because the builder still exposes layout parameters and old loaders.

- [ ] **Step 3: Implement the unified batch API**

Import `LITERATURE_TYPE_CONFIGS` from `database.literature_types`. Remove local `REACTION_CONFIGS` and `CATEGORY_CONFIGS`. Change the function signature to:

```python
def build_vector_databases_batch(
    config_path: str = "./config/config.yaml",
    data_dir: str = "./data/raw",
    literature_type_configs: Optional[Dict[str, Dict]] = None,
    agent_names: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_batch_size: int = 10,
    embedding_concurrency: int = 1,
    max_workers: int = 4,
    sleep_between_batches: float = 0.5,
    resume: bool = True,
    clear_existing: Optional[bool] = None,
    skip_if_exists: bool = False,
    continue_on_error: bool = True,
) -> Dict[str, Dict[str, object]]:
```

Default `literature_type_configs` to the shared constant, remove layout validation/logging, and load once with:

```python
documents = processor.load_literature_type_documents(
    base_dir=data_dir,
    literature_type_configs=literature_type_configs,
)
```

When the root data directory is missing, log every configured `data_dir/path/*.md` and `metadata_csv`, then return `{}`.

- [ ] **Step 4: Remove the batch CLI layout option**

Delete the `--input-layout` argument and pass `literature_type_configs=LITERATURE_TYPE_CONFIGS` from `main()`.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m unittest test.test_build_vector_db_batch_literature_types -v`

Expected: PASS.

Run: `python -m unittest discover -s test -p "test_*.py"`

Expected: PASS with no failures.

- [ ] **Step 6: Commit**

```powershell
git add build_vector_db_batch.py database/text_processor.py test/test_text_processor_doi_extraction.py test/test_build_vector_db_batch_input_layout.py test/test_build_vector_db_batch_literature_types.py
git commit -m "refactor: unify batch literature ingestion"
```

### Task 4: Unify The Legacy Single-Agent Builder

**Files:**
- Create: `test/test_build_vector_db_literature_types.py`
- Modify: `build_vector_db.py`

- [ ] **Step 1: Write failing legacy-builder tests**

Use the existing builder test pattern with patched `AgentConfig`, `TextProcessor`, and `setup_logging`. Make the unified loader return `[]` so the builder exits before embedding. Assert:

```python
build.build_vector_database(
    data_dir=str(data_dir),
    literature_type_configs=build.LITERATURE_TYPE_CONFIGS,
    agent_name="agent1",
)
processor.load_literature_type_documents.assert_called_once_with(
    base_dir=str(data_dir),
    literature_type_configs=build.LITERATURE_TYPE_CONFIGS,
)
parameters = inspect.signature(build.build_vector_database).parameters
self.assertIn("literature_type_configs", parameters)
self.assertNotIn("reaction_configs", parameters)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest test.test_build_vector_db_literature_types -v`

Expected: FAIL because `build_vector_database` still accepts `reaction_configs` and calls the reaction loader.

- [ ] **Step 3: Implement the unified legacy-builder API**

Import the shared constant, remove the local `REACTION_CONFIGS`, rename the argument to `literature_type_configs`, default it to the shared constant, and call:

```python
documents = processor.load_literature_type_documents(
    base_dir=data_dir,
    literature_type_configs=literature_type_configs,
)
```

Update the missing-root diagnostic loop to print each Markdown directory and CSV path. Update the `__main__` call to pass `literature_type_configs=LITERATURE_TYPE_CONFIGS`.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest test.test_build_vector_db_literature_types -v`

Expected: PASS.

Run: `python -m unittest discover -s test -p "test_*.py"`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```powershell
git add build_vector_db.py test/test_build_vector_db_literature_types.py
git commit -m "refactor: unify legacy literature ingestion"
```

### Task 5: Update Documentation And Verify The Repository

**Files:**
- Modify: `README.md`
- Modify: `Developer.md`

- [ ] **Step 1: Update user-facing ingestion documentation**

Document that every Literature Type needs:

```text
data/raw/<literature-type>/*.md
metadata/<configured-name>.csv
```

Document the fixed CSV schema `file_name,doi,abstract`, PDF-to-Markdown basename matching, CSV-to-Markdown-to-`no-doi` fallback, and the unchanged downstream `reaction_type` field. Remove all `--input-layout`, reaction/category layout, XLSX, and `REACTION_CONFIGS` maintenance references from the affected sections.

- [ ] **Step 2: Run stale-reference and formatting checks**

Run:

```powershell
rg -n "REACTION_CONFIGS|CATEGORY_CONFIGS|input_layout|--input-layout|metadata_xlsx|load_reaction_documents|load_category_documents" build_vector_db.py build_vector_db_batch.py database test README.md Developer.md
```

Expected: no matches.

Run: `git diff --check`

Expected: exit code 0 with no output.

- [ ] **Step 3: Run focused ingestion tests**

Run:

```powershell
python -m unittest test.test_literature_type_configs test.test_text_processor_doi_extraction test.test_build_vector_db_batch_literature_types test.test_build_vector_db_literature_types -v
```

Expected: PASS with no failures.

- [ ] **Step 4: Run the complete test suite**

Run: `python -m unittest discover -s test -p "test_*.py"`

Expected: PASS with no failures.

- [ ] **Step 5: Review the final diff against the design**

Run: `git diff --stat` and `git diff --check`, then verify every requirement in `docs/superpowers/specs/2026-07-13-literature-type-csv-design.md` is represented by code, tests, or documentation.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md Developer.md
git commit -m "docs: explain literature type csv ingestion"
```
