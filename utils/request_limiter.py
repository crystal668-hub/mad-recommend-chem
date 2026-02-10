"""
Global request limiter (process-wide).

This module provides a small, thread-safe concurrency cap for outbound network calls
(LLM chat + embeddings). It is intentionally parameter-free (hard-coded defaults) so
it can be enabled without changing CLI flags or config.yaml.

Motivation:
- When running multiple reactions in parallel, transient upstream instability can
  manifest as empty embedding responses (e.g., "No embedding data received") or
  connection flakiness. Capping total in-flight requests reduces burstiness.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from utils.logger import Logger

logger = Logger.create_module_logger("utils.request_limiter")

# Default total in-flight network requests across the whole process.
# Hard-coded on purpose to avoid introducing new configuration knobs.
MAX_INFLIGHT = 6


@dataclass(frozen=True)
class RequestToken:
    kind: str
    acquired_at: float


class GlobalRequestLimiter:
    """
    Thread-safe global limiter.

    Implementation:
    - Uses a BoundedSemaphore to cap total in-flight slots.
    - Tracks an in-flight counter for observability only.
    """

    def __init__(self, max_inflight: int = MAX_INFLIGHT) -> None:
        try:
            cap = int(max_inflight)
        except Exception:
            cap = MAX_INFLIGHT
        cap = max(1, cap)
        self.max_inflight = cap

        self._sem = threading.BoundedSemaphore(value=self.max_inflight)
        self._lock = threading.Lock()
        self._inflight = 0

    @property
    def inflight(self) -> int:
        with self._lock:
            return int(self._inflight)

    def acquire(self, kind: str, timeout: Optional[float] = None) -> RequestToken:
        k = (kind or "unknown").strip() or "unknown"
        start = time.time()

        if timeout is None:
            ok = self._sem.acquire()
        else:
            try:
                t = float(timeout)
            except Exception:
                t = None
            ok = self._sem.acquire(timeout=t) if t is not None else self._sem.acquire()

        if not ok:
            waited = time.time() - start
            logger.debug(
                "request_limiter_acquire_timeout",
                extra={
                    "event": "request_limiter.acquire.timeout",
                    "kind": k,
                    "waited_seconds": waited,
                    "max_inflight": self.max_inflight,
                    "inflight": self.inflight,
                },
            )
            raise TimeoutError(f"Timed out acquiring request limiter slot (kind={k}, timeout={timeout})")

        waited = time.time() - start
        with self._lock:
            self._inflight += 1
            inflight_now = self._inflight

        # Only log when we actually had to wait (to avoid noise at INFO level).
        if waited > 0.5:
            logger.debug(
                "request_limiter_wait",
                extra={
                    "event": "request_limiter.acquire.wait",
                    "kind": k,
                    "waited_seconds": waited,
                    "inflight": inflight_now,
                    "max_inflight": self.max_inflight,
                },
            )

        return RequestToken(kind=k, acquired_at=time.time())

    def release(self, token: RequestToken) -> None:
        # Release the semaphore first; if this raises, keep inflight unchanged so we can detect bugs.
        self._sem.release()
        with self._lock:
            self._inflight = max(0, int(self._inflight) - 1)

    @contextmanager
    def slot(self, kind: str, timeout: Optional[float] = None) -> Iterator[None]:
        token = self.acquire(kind=kind, timeout=timeout)
        try:
            yield
        finally:
            self.release(token)


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_LIMITER: Optional[GlobalRequestLimiter] = None


def get_global_limiter() -> GlobalRequestLimiter:
    """
    Return a process-wide shared limiter instance.

    Notes:
    - This is intentionally a singleton so all threads/reactions/agents share one budget.
    - The limiter is lightweight and safe to import before logging is configured.
    """

    global _GLOBAL_LIMITER
    if _GLOBAL_LIMITER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_LIMITER is None:
                _GLOBAL_LIMITER = GlobalRequestLimiter(max_inflight=MAX_INFLIGHT)
    return _GLOBAL_LIMITER

