from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import requests


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _tei_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _qualified(tag: str, namespace: str) -> str:
    return f"{{{namespace}}}{tag}" if namespace else tag


def extract_profile_xml_segments(profile_xml_text: str, *, max_header_chars: int = 4000, max_body_chars: int = 12000) -> dict[str, str]:
    root = ET.fromstring(profile_xml_text)
    namespace = _tei_namespace(root.tag)
    header = root.find(_qualified("teiHeader", namespace))
    text_node = root.find(_qualified("text", namespace))
    body = text_node.find(_qualified("body", namespace)) if text_node is not None else None
    header_text = _compact_text(" ".join(header.itertext()))[: max_header_chars] if header is not None else ""
    body_text = _compact_text(" ".join(body.itertext()))[: max_body_chars] if body is not None else ""
    return {"header_text": header_text, "body_text": body_text}


class ProfileBuilder:
    def __init__(self, *, grobid_url: str, timeout_seconds: float = 45.0) -> None:
        self.grobid_url = _compact_text(grobid_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def build(self, *, paper_id: str, pdf_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        source_path = Path(pdf_path)
        if not source_path.exists() or source_path.suffix.lower() != ".pdf":
            raise ValueError(f"paper_id={paper_id} does not have a readable local PDF")
        health_response = requests.get(f"{self.grobid_url}/api/isalive", timeout=2.0)
        if int(getattr(health_response, "status_code", 500)) >= 300 or _compact_text(getattr(health_response, "text", "")) != "true":
            raise RuntimeError(f"GROBID server unavailable at {self.grobid_url}")
        with source_path.open("rb") as handle:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={"input": (source_path.name, handle, "application/pdf")},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        xml_text = response.text
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        xml_path = output_root / f"{paper_id}.profile.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        segments = extract_profile_xml_segments(xml_text)
        return {
            "paper_id": paper_id,
            "profile_status": "ready",
            "profile_xml_artifact_path": str(xml_path),
            "profile_header_text": segments["header_text"],
            "profile_body_text": segments["body_text"],
        }


class ListwiseReranker:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = _compact_text(base_url).rstrip("/")
        self.api_key = _compact_text(api_key)
        self.model = _compact_text(model)
        self.timeout_seconds = float(timeout_seconds)
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("LLM config requires base_url, api_key, and model")

    def rerank(self, *, question: str, candidates: list[dict[str, Any]], max_candidates: int) -> dict[str, Any]:
        prompt = {
            "question": question,
            "instructions": (
                "Return strict JSON with a top-level `decisions` list. "
                "Each item must contain `paper_id`, `decision` (`lock` or `drop`), and `reason`."
            ),
            "candidates": candidates,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You are a paper reranker. Return only strict JSON."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
        parsed = json.loads(content)
        decisions = list(parsed.get("decisions") or [])
        if not decisions:
            raise RuntimeError("LLM returned no valid rerank decisions")
        allowed_ids = {str(item.get("paper_id") or "").strip() for item in candidates}
        ranked_candidates: list[dict[str, Any]] = []
        locked_paper_ids: list[str] = []
        dropped_paper_ids: list[str] = []
        decision_map: dict[str, dict[str, Any]] = {}
        for item in decisions:
            paper_id = str(item.get("paper_id") or "").strip()
            decision = str(item.get("decision") or "").strip().lower()
            reason = _compact_text(item.get("reason"))
            if paper_id in allowed_ids and decision in {"lock", "drop"} and reason:
                decision_map[paper_id] = {"paper_id": paper_id, "decision": decision, "reason": reason}
        if not decision_map:
            raise RuntimeError("LLM decisions were invalid after validation")
        for candidate in candidates:
            paper_id = str(candidate.get("paper_id") or "").strip()
            decision = decision_map.get(paper_id, {"paper_id": paper_id, "decision": "drop", "reason": "No valid decision returned."})
            ranked = dict(candidate)
            ranked.update(decision)
            ranked_candidates.append(ranked)
            if decision["decision"] == "lock" and paper_id not in locked_paper_ids and len(locked_paper_ids) < max_candidates:
                locked_paper_ids.append(paper_id)
            elif paper_id not in dropped_paper_ids:
                dropped_paper_ids.append(paper_id)
        return {
            "locked_paper_ids": locked_paper_ids,
            "dropped_paper_ids": dropped_paper_ids,
            "ranked_candidates": ranked_candidates,
            "screen_status": "ready" if locked_paper_ids else "no_locks",
            "failure_domain": "" if locked_paper_ids else "topic_mismatch",
        }


def rerank_candidates(*, request: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = list(request.get("candidates") or [])
    if not candidates:
        raise ValueError("rerank request requires candidates")
    grobid_config = dict(request.get("grobid") or {})
    llm_config = dict(request.get("llm") or {})
    profile_builder = ProfileBuilder(grobid_url=grobid_config.get("url") or "http://localhost:8070")
    paper_profiles: list[dict[str, Any]] = []
    ranked_inputs: list[dict[str, Any]] = []
    for item in candidates:
        paper_id = _compact_text(item.get("paper_id")) or "paper"
        profile = profile_builder.build(paper_id=paper_id, pdf_path=item.get("pdf_path"), output_dir=output_root / "profiles")
        paper_profiles.append(profile)
        ranked_inputs.append(
            {
                "paper_id": paper_id,
                "title": item.get("title"),
                "doi": item.get("doi"),
                "year": item.get("year"),
                "venue": item.get("venue"),
                "retrieval_score": item.get("retrieval_score"),
                "profile_header_text": profile["profile_header_text"],
                "profile_body_text": profile["profile_body_text"],
            }
        )
    api_key = llm_config.get("api_key") or os.environ.get(str(llm_config.get("api_key_env") or "OPENAI_API_KEY"))
    reranker = ListwiseReranker(
        base_url=llm_config.get("base_url") or "https://api.openai.com/v1",
        api_key=api_key or "",
        model=llm_config.get("model") or "",
    )
    rerank_result = reranker.rerank(
        question=_compact_text(request.get("question")),
        candidates=ranked_inputs,
        max_candidates=max(1, int(request.get("max_candidates", 3) or 3)),
    )
    result = {
        **rerank_result,
        "paper_profiles": paper_profiles,
    }
    (output_root / "rerank_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable paper rerank skill")
    parser.add_argument("--request-json", required=True, help="Path to request JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for emitted artifacts")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    result = rerank_candidates(request=request, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
