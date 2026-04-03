# Paper Retrieval Contract

## Providers

- OpenAlex
- Semantic Scholar
- Crossref

## Inputs

- `query_text`
- optional `must_terms`
- optional `exclude_terms`
- optional `year_from` / `year_to`
- optional `preferred_sources`
- optional `limit`

## Output JSON

- `papers`
- `diagnostics`
- `provider_health`
- `request`

Each paper includes normalized `paper_id`, `doi`, `title`, `abstract`, `authors`, `year`, `venue`, `provider_hits`, `retrieval_score`, and OA hints when present.

