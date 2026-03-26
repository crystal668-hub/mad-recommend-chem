from __future__ import annotations

import copy
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from qa.artifacts import QAArtifactStore
from qa.retrieval_state import PaperProfile, PaperRecord


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _tei_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _qualified(tag: str, namespace: str) -> str:
    if namespace:
        return f"{{{namespace}}}{tag}"
    return tag


def extract_profile_xml_segments(
    profile_xml_artifact_path: str,
    *,
    max_header_chars: int = 4000,
    max_body_chars: int = 12000,
) -> Dict[str, str]:
    xml_text = Path(profile_xml_artifact_path).read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    namespace = _tei_namespace(root.tag)
    header = root.find(_qualified("teiHeader", namespace))
    text_node = root.find(_qualified("text", namespace))
    body = text_node.find(_qualified("body", namespace)) if text_node is not None else None
    header_text = _compact_text(" ".join(header.itertext()))[: max(200, int(max_header_chars or 4000))] if header is not None else ""
    body_text = _compact_text(" ".join(body.itertext()))[: max(500, int(max_body_chars or 12000))] if body is not None else ""
    return {
        "header_text": header_text,
        "body_text": body_text,
    }


class GrobidPaperProfileBuilder:
    def __init__(
        self,
        *,
        grobid_url: str = "http://localhost:8070",
        tei_xml_factory: Optional[Callable[[Path], Any]] = None,
        loader_factory: Optional[Callable[[Path], Any]] = None,
        request_post: Optional[Callable[..., Any]] = None,
        request_get: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 45.0,
        preflight_enabled: bool = True,
        startup_enabled: bool = False,
        startup_script: Optional[str] = None,
        startup_timeout_seconds: float = 180.0,
        startup_wait_timeout_seconds: float = 120.0,
        startup_poll_interval_seconds: float = 2.0,
        command_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.grobid_url = str(grobid_url or "http://localhost:8070").strip() or "http://localhost:8070"
        self.tei_xml_factory = tei_xml_factory or loader_factory
        self.request_post = request_post or requests.post
        self.request_get = request_get or requests.get
        self.timeout_seconds = max(1.0, float(timeout_seconds or 45.0))
        self.preflight_enabled = bool(preflight_enabled)
        self.startup_enabled = bool(startup_enabled)
        self.startup_script = str(startup_script or "").strip() or str(
            Path(__file__).resolve().parent.parent / "scripts" / "grobid-up.sh"
        )
        self.startup_timeout_seconds = max(1.0, float(startup_timeout_seconds or 180.0))
        self.startup_wait_timeout_seconds = max(1.0, float(startup_wait_timeout_seconds or 120.0))
        self.startup_poll_interval_seconds = max(0.1, float(startup_poll_interval_seconds or 2.0))
        self.command_runner = command_runner or subprocess.run
        self.last_preflight_payload: Dict[str, Any] = {}

    def build(self, *, paper_record: PaperRecord, artifact_store: Optional[QAArtifactStore] = None) -> PaperProfile:
        store = artifact_store or QAArtifactStore()
        source_pdf_path = self._resolve_local_pdf_artifact_path(paper_record)
        if source_pdf_path is None:
            raise ValueError(f"paper_id={paper_record.paper_id} does not have a readable PDF artifact for GROBID parsing.")

        self._assert_grobid_available()
        tei_xml = self._generate_tei_xml(source_pdf_path)
        clipped_xml = self._clip_tei_xml(tei_xml=tei_xml, paper_id=paper_record.paper_id)
        profile_xml_artifact_path = store.write_text(
            f"proposer_profiles/{paper_record.paper_id}.profile.xml",
            clipped_xml,
        )
        return PaperProfile(
            paper_id=paper_record.paper_id,
            title=paper_record.title,
            doi=paper_record.doi,
            year=paper_record.year,
            venue=paper_record.venue,
            source_artifact_path=str(source_pdf_path),
            profile_status="ready",
            profile_xml_artifact_path=profile_xml_artifact_path,
        )

    def _resolve_local_pdf_artifact_path(self, paper_record: PaperRecord) -> Optional[Path]:
        candidates = [
            str(paper_record.source_artifact_path or "").strip(),
            str(paper_record.fulltext_artifact_path or "").strip(),
        ]
        for candidate_path in candidates:
            if not candidate_path:
                continue
            path = Path(candidate_path)
            if path.exists() and path.suffix.lower() == ".pdf":
                return path
        return None

    def _assert_grobid_available(self) -> None:
        if self.tei_xml_factory is not None:
            return
        probe = self._probe_health(timeout_seconds=2.0)
        if probe.get("available"):
            return
        raise RuntimeError(
            f"GROBID server unavailable at {self.grobid_url}; cannot build proposer XML profiles."
        )

    def ensure_service_available(self) -> Dict[str, Any]:
        if self.tei_xml_factory is not None or not self.preflight_enabled:
            payload = {
                "status": "skipped",
                "grobid_url": self.grobid_url,
                "startup_attempted": False,
                "reason": "preflight disabled or local TEI factory in use",
            }
            self.last_preflight_payload = dict(payload)
            return payload

        initial_probe = self._probe_health(timeout_seconds=2.0)
        if initial_probe.get("available"):
            payload = {
                "status": "healthy",
                "grobid_url": self.grobid_url,
                "startup_attempted": False,
                "health_check": initial_probe,
            }
            self.last_preflight_payload = dict(payload)
            return payload

        payload: Dict[str, Any] = {
            "status": "unavailable",
            "grobid_url": self.grobid_url,
            "startup_attempted": False,
            "health_check": initial_probe,
            "startup_enabled": self.startup_enabled,
            "startup_script": self.startup_script,
        }
        self.last_preflight_payload = dict(payload)
        if not self.startup_enabled:
            raise RuntimeError(
                f"GROBID server unavailable at {self.grobid_url}; preflight failed before react_reviewed execution."
            )

        startup_result = self._run_startup_script()
        deadline = time.perf_counter() + self.startup_wait_timeout_seconds
        final_probe = initial_probe
        while time.perf_counter() < deadline:
            final_probe = self._probe_health(timeout_seconds=2.0)
            if final_probe.get("available"):
                payload = {
                    "status": "healthy",
                    "grobid_url": self.grobid_url,
                    "startup_attempted": True,
                    "health_check": final_probe,
                    "startup_enabled": self.startup_enabled,
                    "startup_script": self.startup_script,
                    "startup_result": startup_result,
                }
                self.last_preflight_payload = dict(payload)
                return payload
            time.sleep(self.startup_poll_interval_seconds)

        payload = {
            "status": "unavailable",
            "grobid_url": self.grobid_url,
            "startup_attempted": True,
            "health_check": final_probe,
            "startup_enabled": self.startup_enabled,
            "startup_script": self.startup_script,
            "startup_result": startup_result,
        }
        self.last_preflight_payload = dict(payload)
        raise RuntimeError(
            f"GROBID server unavailable at {self.grobid_url}; startup was attempted but the service did not become healthy in time."
        )

    def _probe_health(self, *, timeout_seconds: float) -> Dict[str, Any]:
        health_url = f"{self.grobid_url.rstrip('/')}/api/isalive"
        try:
            response = self.request_get(health_url, timeout=timeout_seconds)
            status_code = int(getattr(response, "status_code", 200))
            body_text = _compact_text(getattr(response, "text", "") or "")
            available = 200 <= status_code < 300 and body_text.lower() == "true"
            return {
                "available": available,
                "url": health_url,
                "status_code": status_code,
                "body": body_text[:200],
            }
        except Exception as exc:
            return {
                "available": False,
                "url": health_url,
                "error": _compact_text(str(exc)) or exc.__class__.__name__,
            }

    def _run_startup_script(self) -> Dict[str, Any]:
        startup_path = Path(self.startup_script)
        if not startup_path.is_absolute():
            startup_path = (Path(__file__).resolve().parent.parent / startup_path).resolve()
        if not startup_path.exists():
            raise RuntimeError(
                f"GROBID startup script is missing at {startup_path}; cannot auto-start the service."
            )
        result = self.command_runner(
            ["/bin/bash", str(startup_path)],
            capture_output=True,
            text=True,
            timeout=self.startup_timeout_seconds,
            check=False,
        )
        payload = {
            "returncode": int(getattr(result, "returncode", 1) or 0),
            "stdout": _compact_text(getattr(result, "stdout", "") or "")[:500],
            "stderr": _compact_text(getattr(result, "stderr", "") or "")[:500],
        }
        if payload["returncode"] != 0:
            raise RuntimeError(
                f"GROBID startup script failed with exit code {payload['returncode']} at {startup_path}."
            )
        return payload

    def _generate_tei_xml(self, source_path: Path) -> str:
        if self.tei_xml_factory is not None:
            payload = self.tei_xml_factory(source_path)
            if isinstance(payload, bytes):
                return payload.decode("utf-8", errors="ignore")
            return str(payload)

        with source_path.open("rb") as handle:
            response = self.request_post(
                f"{self.grobid_url.rstrip('/')}/api/processFulltextDocument",
                files={"input": (source_path.name, handle, "application/pdf")},
                headers={"Accept": "application/xml"},
                timeout=self.timeout_seconds,
            )
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise RuntimeError(f"GROBID returned HTTP {status_code} for {source_path.name}.")
        tei_xml = str(getattr(response, "text", "") or "").strip()
        if not tei_xml:
            raise RuntimeError(f"GROBID returned empty TEI XML for {source_path.name}.")
        return tei_xml

    def _clip_tei_xml(self, *, tei_xml: str, paper_id: str) -> str:
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:
            raise RuntimeError(f"paper_id={paper_id} produced invalid TEI XML from GROBID.") from exc

        namespace = _tei_namespace(root.tag)
        if namespace:
            ET.register_namespace("", namespace)

        tei_header = root.find(_qualified("teiHeader", namespace))
        text_node = root.find(_qualified("text", namespace))
        body = text_node.find(_qualified("body", namespace)) if text_node is not None else None
        if tei_header is None:
            raise RuntimeError(f"paper_id={paper_id} TEI XML is missing teiHeader.")
        if body is None:
            raise RuntimeError(f"paper_id={paper_id} TEI XML is missing text/body.")

        clipped_root = ET.Element(root.tag, dict(root.attrib))
        clipped_root.append(copy.deepcopy(tei_header))
        clipped_text = ET.SubElement(clipped_root, _qualified("text", namespace))
        clipped_text.append(copy.deepcopy(body))
        return ET.tostring(clipped_root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def write_profile_failure(
    *,
    store: QAArtifactStore,
    paper_record: PaperRecord,
    reason: str,
) -> str:
    payload = {
        "paper_id": paper_record.paper_id,
        "title": paper_record.title,
        "doi": paper_record.doi,
        "year": paper_record.year,
        "venue": paper_record.venue,
        "source_artifact_path": paper_record.source_artifact_path,
        "reason": _compact_text(reason) or "unknown profile extraction error",
    }
    return store.write_json(f"proposer_profiles/{paper_record.paper_id}.profile_failure.json", payload)
