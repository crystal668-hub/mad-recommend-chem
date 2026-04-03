# Paper Rerank Contract

## Request Shape

```json
{
  "question": "Which papers best support the HER claim?",
  "max_candidates": 3,
  "grobid": {
    "url": "http://localhost:8070"
  },
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini"
  },
  "candidates": [
    {
      "paper_id": "paper-1",
      "title": "Paper 1",
      "retrieval_score": 7.2,
      "pdf_path": "/tmp/paper-1.pdf"
    }
  ]
}
```

## Output JSON

- `locked_paper_ids`
- `dropped_paper_ids`
- `ranked_candidates`
- `paper_profiles`
- `screen_status`
- `failure_domain`

## Failure Policy

- Missing readable PDF: fail that candidate explicitly
- GROBID unavailable: hard fail
- LLM config missing or invalid: hard fail
- Invalid or empty structured decisions: hard fail

