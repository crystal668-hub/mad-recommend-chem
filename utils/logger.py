"""
Logging utilities for MAD.

Design goals
- Configure logging once per process (via `setup_logging`).
- Avoid per-module log files / timestamped logger names.
- Group logs by run_id under `./logs/runs/<run_id>/`.
- Keep backwards-compatible APIs (`Logger.get_logger`, `DebateLogger`).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# -------------------------
# Run-scoped context
# -------------------------

_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("mad_run_id", default="")
_RUN_DIR: Optional[Path] = None

_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()


def get_run_id() -> str:
    return _RUN_ID.get() or ""


def get_run_dir() -> Optional[Path]:
    return _RUN_DIR


def _set_run_context(run_id: str, run_dir: Path) -> None:
    _RUN_ID.set(run_id)
    global _RUN_DIR
    _RUN_DIR = run_dir


def make_debate_id(engine: str, components: Any, reaction_type: Any) -> str:
    """
    Create a unique, distinguishable debate id.

    Format:
      <engine>_<UTC timestamp>_<hash8>_<rand6>
    """
    eng = (engine or "unknown").strip().lower()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "engine": eng,
        "components": components,
        "reaction_type": reaction_type,
        "ts": ts,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:8]
    rand = uuid.uuid4().hex[:6]
    return f"{eng}_{ts}_{digest}_{rand}"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_debate_artifacts(
    debate_id: str,
    engine: str,
    payload: Dict[str, Any],
    transcript_events: Any,
) -> Dict[str, str]:
    """
    Write one full-debate structured file + one transcript jsonl file.

    Returns:
      {"full_path": "...", "transcript_path": "..."}
    """
    base = get_run_dir() or Path("./logs")
    debates_dir = _ensure_dir(base / "debates")

    safe_id = (debate_id or "debate").strip()
    full_path = debates_dir / f"debate_{safe_id}.json"
    transcript_path = debates_dir / f"debate_{safe_id}_transcript.jsonl"

    # Full payload: one JSON.
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=str)

    # Transcript: one JSON object per line for easy streaming/grep.
    with open(transcript_path, "w", encoding="utf-8") as f:
        if isinstance(transcript_events, list):
            for idx, evt in enumerate(transcript_events):
                if not isinstance(evt, dict):
                    evt = {"value": evt}
                line = {
                    "event_index": idx,
                    "debate_id": safe_id,
                    "engine": engine,
                    "run_id": get_run_id() or None,
                    **evt,
                }
                f.write(json.dumps(line, ensure_ascii=True, default=str) + "\n")
        else:
            # Fallback: dump a single line.
            f.write(
                json.dumps(
                    {
                        "event_index": 0,
                        "debate_id": safe_id,
                        "engine": engine,
                        "run_id": get_run_id() or None,
                        "value": transcript_events,
                    },
                    ensure_ascii=True,
                    default=str,
                )
                + "\n"
            )

    return {"full_path": str(full_path), "transcript_path": str(transcript_path)}


# -------------------------
# Formatters / Filters
# -------------------------


class _ContextFilter(logging.Filter):
    """Injects run_id so formatters can always reference it."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - record is stdlib name
        record.run_id = get_run_id() or "-"  # type: ignore[attr-defined]
        return True


class _PrefixFilter(logging.Filter):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - record is stdlib name
        return bool(record.name and record.name.startswith(self.prefix))


_STD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line (jsonl), including `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - record is stdlib name
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "msg": record.getMessage(),
        }

        # Preserve exception info if present.
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Collect non-standard attributes (from `extra=`).
        for k, v in record.__dict__.items():
            if k in _STD_ATTRS or k.startswith("_"):
                continue
            # Ensure JSON-serializable (fall back to string).
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        # Use ensure_ascii=True so logs are stable across terminals/encodings.
        return json.dumps(payload, ensure_ascii=True)


# -------------------------
# Public API
# -------------------------


class Logger:
    """Compatibility wrapper around `logging.getLogger`."""

    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(
        cls,
        name: str = "MAD",
        log_file: Optional[str] = None,
        level: str = "INFO",
        log_format: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Get a logger instance.

        Notes:
        - If `setup_logging()` has already configured process-wide handlers, this function will
          *not* attach per-logger handlers (to avoid duplicate logs).
        - If logging is not configured yet and `log_file` is provided, we attach handlers as a
          fallback for standalone scripts.
        """
        logger = logging.getLogger(name)

        # Fast path once configured: rely on root handlers.
        if _CONFIGURED:
            return logger

        # Side-effect-free by default: if no log_file is requested, just return the logger.
        # This prevents import-time handler creation in library modules.
        if not log_file:
            return logger

        # Backwards-compatible fallback for standalone scripts/tests that request a file.
        if logger.handlers:
            return logger

        try:
            logger.setLevel(getattr(logging, level.upper()))
        except Exception:
            logger.setLevel(logging.INFO)

        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Avoid double logging if later a root logger is configured.
        logger.propagate = False
        cls._loggers[name] = logger
        return logger

    @classmethod
    def create_module_logger(cls, module_name: str) -> logging.Logger:
        # Must be side-effect-free even before setup_logging().
        return logging.getLogger(f"MAD.{module_name}")


def setup_logging(config: dict, run_id: Optional[str] = None) -> logging.Logger:
    """
    Configure process-wide logging (root handlers) and return the main "MAD" logger.

    Output layout (defaults):
      logs/
        system.log                 # rolling "latest" file (compatible with existing config)
        runs/<run_id>/
          run.log                  # full run log (text)
          events.jsonl             # full run log (structured)
          debate.log               # filtered (MAD.debate.*)
          db.log                   # filtered (MAD.database.*)
    """
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return logging.getLogger("MAD")

        log_config = (config or {}).get("logging", {}) or {}
        level_name = str(log_config.get("level", "INFO")).upper()
        log_level = getattr(logging, level_name, logging.INFO)

        log_file = str(log_config.get("log_file", "./logs/system.log"))
        log_format = str(
            log_config.get(
                "log_format",
                "%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s",
            )
        )
        max_file_size = int(log_config.get("max_file_size", 10 * 1024 * 1024))
        backup_count = int(log_config.get("backup_count", 5))

        run_dir_root = Path(str(log_config.get("run_dir", "./logs/runs")))
        run_id_final = (run_id or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
        run_dir = run_dir_root / run_id_final
        run_dir.mkdir(parents=True, exist_ok=True)
        _set_run_context(run_id_final, run_dir)

        # Root logger: one place to attach handlers.
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)  # handlers decide what to emit

        # Remove existing handlers to avoid duplicates (e.g., from prior tests).
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

        formatter = logging.Formatter(log_format)
        ctx_filter = _ContextFilter()

        # Console (human) — INFO+
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        console.addFilter(ctx_filter)
        root.addHandler(console)

        # Rolling "latest" file (compatible with existing config.yaml)
        latest_path = Path(log_file)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest = RotatingFileHandler(
            str(latest_path),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8",
        )
        latest.setLevel(logging.DEBUG)
        latest.setFormatter(formatter)
        latest.addFilter(ctx_filter)
        root.addHandler(latest)

        # Per-run full log
        run_log = logging.FileHandler(str(run_dir / "run.log"), encoding="utf-8")
        run_log.setLevel(logging.DEBUG)
        run_log.setFormatter(formatter)
        run_log.addFilter(ctx_filter)
        root.addHandler(run_log)

        # Per-run structured log (jsonl)
        events = logging.FileHandler(str(run_dir / "events.jsonl"), encoding="utf-8")
        events.setLevel(logging.DEBUG)
        events.setFormatter(JsonLineFormatter())
        events.addFilter(ctx_filter)
        root.addHandler(events)

        # Per-run debate/db convenience logs
        debate = logging.FileHandler(str(run_dir / "debate.log"), encoding="utf-8")
        debate.setLevel(logging.DEBUG)
        debate.setFormatter(formatter)
        debate.addFilter(ctx_filter)
        debate.addFilter(_PrefixFilter("MAD.debate"))
        root.addHandler(debate)

        db = logging.FileHandler(str(run_dir / "db.log"), encoding="utf-8")
        db.setLevel(logging.DEBUG)
        db.setFormatter(formatter)
        db.addFilter(ctx_filter)
        db.addFilter(_PrefixFilter("MAD.database"))
        root.addHandler(db)

        # Reduce noise from very chatty dependencies (can be overridden by user config later).
        for noisy in ["httpx", "httpcore", "openai", "chromadb", "langchain", "langchain_core"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True

        logger = logging.getLogger("MAD")
        logger.info("logging_initialized", extra={"event": "logging.init", "run_id": run_id_final, "run_dir": str(run_dir)})
        return logger


class DebateLogger:
    """
    Backwards-compatible debate logger wrapper.

    The project now routes debate logs into the process-wide handlers. This class exists so older
    code can keep calling `DebateLogger().log_*()` without managing handlers/files.
    """

    def __init__(self) -> None:
        self.debate_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("MAD.debate")

    def log_debate_start(self, components: list, config: dict):
        self.logger.info(
            "debate_start",
            extra={
                "event": "debate.start",
                "debate_id": self.debate_id,
                "components": components,
                "max_rounds": (config or {}).get("max_rounds"),
            },
        )

    def log_round_start(self, round_num: int):
        self.logger.info(
            "debate_round_start",
            extra={"event": "debate.round.start", "debate_id": self.debate_id, "round": int(round_num)},
        )

    def log_agent_response(self, agent_name: str, response: str, products: str = None, performance: str = None):
        self.logger.info(
            "debate_agent_response",
            extra={
                "event": "debate.agent.response",
                "debate_id": self.debate_id,
                "agent_name": agent_name,
                "products": products,
                "performance": performance,
                "response_preview": (response or "")[:500],
            },
        )

    def log_consensus_check(self, consensus: bool, details: str):
        self.logger.info(
            "debate_consensus_check",
            extra={
                "event": "debate.consensus",
                "debate_id": self.debate_id,
                "consensus": bool(consensus),
                "details": (details or "")[:1000],
            },
        )

    def log_debate_end(self, result: dict):
        self.logger.info(
            "debate_end",
            extra={
                "event": "debate.end",
                "debate_id": self.debate_id,
                "consensus_reached": (result or {}).get("consensus_reached"),
                "final_products": (result or {}).get("final_products"),
                "final_performance": (result or {}).get("final_performance"),
                "debate_rounds": (result or {}).get("debate_rounds"),
                "time_elapsed": (result or {}).get("time_elapsed"),
                "engine": (result or {}).get("engine"),
            },
        )

    def get_log_file_path(self) -> str:
        # Prefer the per-run debate.log if available.
        run_dir = get_run_dir()
        if run_dir:
            return str(run_dir / "debate.log")
        return ""
