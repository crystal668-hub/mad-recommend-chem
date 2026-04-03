from __future__ import annotations

import argparse
import json
import mimetypes
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests


DEFAULT_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,text/html;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def normalize_doi(value: Optional[str]) -> Optional[str]:
    text = _compact_text(value)
    if not text:
        return None
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.lower() or None


def _guess_extension(content_type: str) -> str:
    if content_type == "application/pdf":
        return ".pdf"
    return mimetypes.guess_extension(content_type) or ".bin"


@dataclass
class FetchedDocument:
    url: str
    content_type: str
    text: Optional[str] = None
    binary: Optional[bytes] = None
    final_url: Optional[str] = None
    redirect_count: int = 0


class _HttpClient:
    def __init__(self, *, timeout: float = 15.0, retry_attempts: int = 2) -> None:
        self.timeout = float(timeout)
        self.retry_attempts = max(0, int(retry_attempts))

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        attempts = self.retry_attempts + 1
        for attempt in range(1, attempts + 1):
            try:
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException:
                if attempt >= attempts:
                    raise
                time.sleep(min(1.0 * (2 ** (attempt - 1)) + random.random() * 0.1, 4.0))
        raise RuntimeError("request failed")


class UnpaywallClient:
    def __init__(self, *, email: Optional[str], timeout: float = 10.0) -> None:
        self.email = _compact_text(email) or None
        self.http = _HttpClient(timeout=timeout)

    def lookup(self, doi: Optional[str]) -> Optional[dict[str, Any]]:
        normalized = normalize_doi(doi)
        if not normalized or not self.email:
            return None
        response = self.http.request(
            "GET",
            f"https://api.unpaywall.org/v2/{quote(normalized, safe='')}",
            params={"email": self.email},
        )
        return response.json()


class HttpTextFetcher:
    def __init__(self, *, timeout: float = 15.0) -> None:
        self.http = _HttpClient(timeout=timeout)

    def fetch(self, url: str, *, headers: Optional[dict[str, str]] = None) -> FetchedDocument:
        request_headers = dict(DEFAULT_BROWSER_HEADERS)
        if headers:
            request_headers.update(headers)
        response = self.http.request("GET", url, headers=request_headers, allow_redirects=True)
        content_type = str(response.headers.get("content-type") or "").split(";")[0].strip().lower() or "application/octet-stream"
        if content_type == "application/pdf" or str(url).lower().endswith(".pdf"):
            return FetchedDocument(url=url, content_type="application/pdf", binary=response.content, final_url=str(response.url), redirect_count=len(response.history or []))
        if content_type.startswith("text/") or content_type in {"application/xml", "application/xhtml+xml", "application/json"}:
            return FetchedDocument(url=url, content_type=content_type, text=response.text, final_url=str(response.url), redirect_count=len(response.history or []))
        return FetchedDocument(url=url, content_type=content_type, binary=response.content, final_url=str(response.url), redirect_count=len(response.history or []))


class PdfUrlProbeClient:
    def __init__(self, *, timeout: float = 5.0, range_bytes: int = 1024) -> None:
        self.http = _HttpClient(timeout=timeout)
        self.range_bytes = max(64, int(range_bytes))

    def probe(self, url: str) -> dict[str, Any]:
        headers = dict(DEFAULT_BROWSER_HEADERS)
        headers["Accept"] = "application/pdf,application/octet-stream,text/html;q=0.8,*/*;q=0.5"
        try:
            head = self.http.request("HEAD", url, headers=headers, allow_redirects=True)
            content_type = str(head.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type == "application/pdf":
                return {"verdict": "strong", "method": "head", "final_url": str(head.url), "content_type": content_type}
        except Exception:
            pass
        response = self.http.request(
            "GET",
            url,
            headers={**headers, "Range": f"bytes=0-{self.range_bytes - 1}"},
            allow_redirects=True,
            stream=True,
        )
        try:
            sample = next(response.iter_content(chunk_size=self.range_bytes), b"")
            content_type = str(response.headers.get("content-type") or "").split(";")[0].strip().lower()
            verdict = "strong" if content_type == "application/pdf" or bytes(sample[:5]) == b"%PDF-" else "non_pdf"
            return {"verdict": verdict, "method": "range_get", "final_url": str(response.url), "content_type": content_type}
        finally:
            response.close()


class AccessEngine:
    def __init__(self, *, unpaywall_email: Optional[str] = None) -> None:
        self.unpaywall = UnpaywallClient(email=unpaywall_email)
        self.fetcher = HttpTextFetcher()
        self.probe = PdfUrlProbeClient()

    def access(self, *, request: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        warnings: list[dict[str, Any]] = []
        documents: list[dict[str, Any]] = []
        for item in list(request.get("documents") or []):
            try:
                documents.append(self._access_one(item=item, request=request, output_dir=destination))
            except Exception as exc:
                warnings.append({"paper_id": item.get("paper_id"), "status": "failure", "error": str(exc)})
        return {"documents": documents, "warnings": warnings}

    def _access_one(self, *, item: dict[str, Any], request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        paper_id = _compact_text(item.get("paper_id")) or "paper"
        doi = normalize_doi(item.get("doi"))
        oa_url = _compact_text(item.get("oa_url")) or None
        if request.get("prefer_unpaywall") and doi:
            unpaywall_payload = self.unpaywall.lookup(doi)
            if unpaywall_payload:
                best_location = dict(unpaywall_payload.get("best_oa_location") or {})
                oa_url = _compact_text(best_location.get("url_for_pdf") or best_location.get("url_for_landing_page") or best_location.get("url")) or oa_url
        if not oa_url:
            raise ValueError(f"paper_id={paper_id} has no usable OA URL")
        probe_result = None
        if request.get("probe_pdf_urls"):
            probe_result = self.probe.probe(oa_url)
        fetched = self.fetcher.fetch(oa_url)
        artifact_extension = _guess_extension(fetched.content_type)
        artifact_path = output_dir / f"{paper_id}{artifact_extension}"
        if fetched.binary is not None:
            artifact_path.write_bytes(fetched.binary)
            fulltext_status = "binary_only"
        else:
            artifact_path.write_text(fetched.text or "", encoding="utf-8")
            fulltext_status = "text_only"
        return {
            "paper_id": paper_id,
            "doi": doi,
            "source_url": oa_url,
            "final_url": fetched.final_url,
            "artifact_path": str(artifact_path),
            "content_type": fetched.content_type,
            "redirect_count": fetched.redirect_count,
            "fulltext_status": fulltext_status,
            "pdf_probe": probe_result,
        }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable paper access skill")
    parser.add_argument("--request-json", required=True, help="Path to the request JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for downloaded artifacts")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    engine = AccessEngine(unpaywall_email=request.get("unpaywall_email"))
    result = engine.access(request=request, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
