---
name: paper-access
description: Use when an agent needs to resolve OA URLs, probe PDF endpoints, and download local paper artifacts from DOI or URL inputs.
---

# Paper Access

## Overview

Resolve paper access paths and download local artifacts from explicit DOI and URL inputs. This skill handles Unpaywall lookups, HTTP fetches, PDF probes, and artifact emission.

## When to Use

Use this skill when:
- a paper candidate already exists and the next step is acquisition
- an agent needs OA resolution from DOI metadata
- a downloader must verify whether a URL really serves a PDF

Do not use this skill for parsing or reranking.

## Execution

```bash
python skill/paper-access/scripts/paper_access.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/paper-access-out
```

Read `references/contracts.md` for request examples and environment variables.

