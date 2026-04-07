# chemqa-review Contract

## Install Layout

Place this bundle under:

```text
<skills-root>/chemqa-review
```

Required sibling bundles:

```text
<skills-root>/debateclaw-v1
<skills-root>/paper-retrieval
<skills-root>/paper-access
<skills-root>/paper-parse
<skills-root>/paper-rerank
```

## Runtime Outputs

The bridge scripts generate:

- file-backed run plans under `control/runplans/`
- generated prompt bundles under `generated/prompt-bundles/`
- generated command maps under `generated/command-maps/`
- generated runtime context under `generated/runtime-context/`

## Post-Run Output

The debate coordinator is expected to provide `chemqa_review_protocol.json`.
`collect_artifacts.py` then rebuilds a `react_reviewed`-compatible artifact set.
