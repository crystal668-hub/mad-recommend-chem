---
name: paper-parse
description: Use when an agent needs to parse a local paper PDF or text artifact into fulltext, sections, snippets, and an extraction report.
---

# Paper Parse

## Overview

Parse a local document into structured text artifacts. This skill is self-contained and assumes only a file path plus optional parser settings.

The default PDF stack is fixed:
- Primary parser: `PyMuPDF`
- Fallback parser: `Docling`

## When to Use

Use this skill when:
- a paper has already been downloaded locally
- the next step needs clean fulltext or section boundaries
- an agent needs portable parsing behavior outside the current ChemQA runtime

Do not use this skill for:
- remote paper search
- OA resolution or HTTP downloading
- GROBID TEI/profile generation for reranking

## Execution

Run the parser script with a local input path and output directory:

```bash
python skill/paper-parse/scripts/paper_parse.py \
  --input /path/to/paper.pdf \
  --output-dir /tmp/paper-parse-out
```

The script writes JSON to stdout and stores artifacts in the output directory.

## Inputs And Outputs

- Input: local `.pdf`, `.txt`, or `.md` path
- Output: normalized `fulltext`, `sections`, `snippets`, extraction report, warnings, and parser metadata

Read `references/contracts.md` for the JSON contract and failure semantics.

## Failure Modes

- Invalid PDF header: hard fail with `fulltext_unusable`
- PyMuPDF extraction rejected by quality gates: automatically try Docling
- Docling also fails: hard fail with explicit reasons and attempt metadata

