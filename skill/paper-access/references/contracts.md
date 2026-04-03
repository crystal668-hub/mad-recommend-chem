# Paper Access Contract

## Request Shape

```json
{
  "documents": [
    {
      "paper_id": "doi-10-1000-example",
      "doi": "10.1000/example",
      "oa_url": "https://example.org/paper.pdf"
    }
  ],
  "prefer_unpaywall": true,
  "probe_pdf_urls": true,
  "unpaywall_email": "name@example.org"
}
```

## Output JSON

- `documents`
- `warnings`

Each document includes local `artifact_path`, `content_type`, `final_url`, `fulltext_status`, and download metadata.

