---
name: paper-rerank
description: Use when an agent needs to build portable paper profiles from local PDFs and run listwise LLM reranking with explicit inputs, without relying on ChemQA workspace state or hidden carry-over data.
---

# Paper Rerank

## Overview

Build profile artifacts from local paper PDFs and ask an LLM to produce `lock` or `drop` rerank decisions. This skill is self-contained and requires explicit candidate inputs plus explicit external service configuration.

## When to Use

Use this skill when:
- candidate papers are already downloaded locally
- reranking should consider more than title or abstract metadata
- the caller can provide GROBID and LLM configuration explicitly

Do not use this skill for remote search, downloading, or fulltext parsing.

## Execution

```bash
python skill/paper-rerank/scripts/paper_rerank.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/paper-rerank-out
```

Read `references/contracts.md` for request fields and required environment values.

