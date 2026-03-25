# GROBID Docker Compose

## Start

```bash
docker compose up -d
```

The service will listen on `http://127.0.0.1:8070`.

The first start will pull `grobid/grobid:0.8.2.1-crf`, so it can take a while.

If your current WSL shell does not expose the Linux `docker` CLI, use:

```bash
./scripts/grobid-up.sh
```

## Check Health

```bash
curl http://127.0.0.1:8070/api/isalive
```

Expected response:

```text
true
```

## Stop

```bash
docker compose stop
```

## Remove

```bash
docker compose down
```

If you started GROBID through `./scripts/grobid-up.sh` in this WSL environment, you can still stop or remove it with the same Compose file through Windows Docker Desktop:

```bash
powershell.exe -NoProfile -Command "docker compose -f compose.yaml stop"
```

```bash
powershell.exe -NoProfile -Command "docker compose -f compose.yaml down"
```

## Parse a PDF

```bash
curl -sS -o outputs/paper.tei.xml \
  -F input=@/absolute/path/to/paper.pdf \
  -F consolidateHeader=0 \
  -F consolidateCitations=0 \
  -F includeRawCitations=1 \
  http://127.0.0.1:8070/api/processFulltextDocument
```

## Notes

- ChemQA currently uses GROBID for `react_reviewed` paper profiling and reranking.
- ChemQA's main PDF full-text extraction path is still `PyMuPDF`, not GROBID.
