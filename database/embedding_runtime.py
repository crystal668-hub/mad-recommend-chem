"""Quota-aware runtime primitives for offline embedding builds."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional, Sequence, TypeVar
from urllib.parse import urlparse


T = TypeVar("T")


DEFAULT_GLOBAL_MAX_INFLIGHT = 28
MYRIMATE_QUOTA_GROUPS = {
    "myrimate",
    "endpoint:agent-team-api.myrimate.cn",
}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    honor_retry_after: bool = True

    def normalized(self) -> "RetryPolicy":
        return RetryPolicy(
            max_attempts=max(1, int(self.max_attempts)),
            base_delay_seconds=max(0.0, float(self.base_delay_seconds)),
            max_delay_seconds=max(0.0, float(self.max_delay_seconds)),
            honor_retry_after=bool(self.honor_retry_after),
        )


@dataclass(frozen=True)
class QuotaPolicy:
    initial_inflight: int = 1
    max_inflight: int = 1
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    cooldown_seconds: float = 30.0
    recovery_successes: int = 100
    window_seconds: float = 60.0

    def normalized(self) -> "QuotaPolicy":
        maximum = max(1, int(self.max_inflight))
        initial = max(1, min(int(self.initial_inflight), maximum))

        def _positive_optional(value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            parsed = int(value)
            return parsed if parsed > 0 else None

        return QuotaPolicy(
            initial_inflight=initial,
            max_inflight=maximum,
            requests_per_minute=_positive_optional(self.requests_per_minute),
            tokens_per_minute=_positive_optional(self.tokens_per_minute),
            cooldown_seconds=max(0.0, float(self.cooldown_seconds)),
            recovery_successes=max(1, int(self.recovery_successes)),
            window_seconds=max(0.01, float(self.window_seconds)),
        )


@dataclass(frozen=True)
class AgentEmbeddingRuntime:
    quota_group: str
    request_batch_size: int
    max_batch_items: int
    max_batch_tokens: Optional[int]


@dataclass
class EmbeddingRuntimeSettings:
    global_max_inflight: int = DEFAULT_GLOBAL_MAX_INFLIGHT
    request_timeout_seconds: float = 60.0
    write_batch_size: int = 100
    write_queue_max_batches: int = 8
    write_flush_interval_ms: int = 500
    existing_id_check_batch_size: int = 1000
    failure_manifest_dir: str = "./outputs/embedding_failures"
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    quota_groups: Dict[str, QuotaPolicy] = field(default_factory=dict)
    agents: Dict[str, AgentEmbeddingRuntime] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        runtime_config: Optional[Mapping[str, Any]],
        agent_configs: Mapping[str, Mapping[str, Any]],
        legacy_batch_size: Optional[int] = None,
        legacy_concurrency: Optional[int] = None,
        write_batch_size: Optional[int] = None,
        global_max_inflight: Optional[int] = None,
    ) -> "EmbeddingRuntimeSettings":
        cfg = dict(runtime_config or {})
        retry_cfg = dict(cfg.get("retry") or {})
        retry_policy = RetryPolicy(
            max_attempts=retry_cfg.get("max_attempts", 6),
            base_delay_seconds=retry_cfg.get("base_delay_seconds", 1),
            max_delay_seconds=retry_cfg.get("max_delay_seconds", 60),
            honor_retry_after=retry_cfg.get("honor_retry_after", True),
        ).normalized()

        raw_groups = dict(cfg.get("quota_groups") or {})
        groups: Dict[str, QuotaPolicy] = {}
        for name, value in raw_groups.items():
            group_cfg = dict(value or {})
            groups[str(name)] = QuotaPolicy(
                initial_inflight=group_cfg.get("initial_inflight", group_cfg.get("max_inflight", 1)),
                max_inflight=group_cfg.get("max_inflight", 1),
                requests_per_minute=group_cfg.get("requests_per_minute"),
                tokens_per_minute=group_cfg.get("tokens_per_minute"),
                cooldown_seconds=group_cfg.get("cooldown_seconds", 30),
                recovery_successes=group_cfg.get("recovery_successes", 100),
                window_seconds=group_cfg.get("window_seconds", 60),
            ).normalized()

        agents: Dict[str, AgentEmbeddingRuntime] = {}
        explicit_groups = bool(raw_groups)
        for agent_name, raw_agent_cfg in agent_configs.items():
            agent_cfg = dict(raw_agent_cfg or {})
            provider = str(agent_cfg.get("embedding_provider") or "openrouter").strip().lower()
            model = str(agent_cfg.get("embedding_model") or "").strip().lower()
            singleton_models = {"google/gemini-embedding-2"}
            configured_group = agent_cfg.get("embedding_quota_group")
            if configured_group:
                quota_group = str(configured_group).strip()
            elif provider == "voyage":
                quota_group = "voyage_org"
            else:
                endpoint = str(agent_cfg.get("emb_url") or agent_cfg.get("base_url") or "")
                hostname = (urlparse(endpoint).hostname or "").strip().lower()
                quota_group = f"endpoint:{hostname}" if hostname else provider
            if explicit_groups and quota_group not in groups:
                raise ValueError(f"Agent '{agent_name}' references unknown quota group '{quota_group}'")
            if quota_group not in groups:
                if provider == "voyage":
                    groups[quota_group] = QuotaPolicy(
                        initial_inflight=2,
                        max_inflight=4,
                        requests_per_minute=2000,
                        tokens_per_minute=3000000,
                    ).normalized()
                elif quota_group.lower() in MYRIMATE_QUOTA_GROUPS:
                    groups[quota_group] = QuotaPolicy(
                        initial_inflight=16,
                        max_inflight=24,
                    ).normalized()
                else:
                    groups[quota_group] = QuotaPolicy(
                        initial_inflight=2,
                        max_inflight=4,
                    ).normalized()

            request_size = legacy_batch_size
            if request_size is None:
                provider_default_batch = (
                    1
                    if model in singleton_models
                    else {
                        "voyage": 128,
                        "aliyun": 1,
                        "zenmux": 32,
                        "openrouter": 32,
                    }.get(provider, 10)
                )
                request_size = agent_cfg.get("embedding_request_batch_size", provider_default_batch)
            request_size = max(1, int(request_size))
            provider_max_items = (
                1
                if model in singleton_models
                else {
                    "zenmux": 2048,
                    "openrouter": 2048,
                    "voyage": 128,
                    "aliyun": 1,
                }.get(provider, request_size)
            )
            max_items = max(1, int(agent_cfg.get("embedding_max_batch_items", provider_max_items)))
            request_size = min(request_size, max_items)
            raw_max_tokens = agent_cfg.get("embedding_max_batch_tokens")
            if raw_max_tokens in (None, "") and provider in {"zenmux", "openrouter"}:
                raw_max_tokens = 300000
            max_tokens = None if raw_max_tokens in (None, "") else max(1, int(raw_max_tokens))
            agents[str(agent_name)] = AgentEmbeddingRuntime(
                quota_group=quota_group,
                request_batch_size=request_size,
                max_batch_items=max_items,
                max_batch_tokens=max_tokens,
            )

        if legacy_concurrency is not None:
            concurrency = max(1, int(legacy_concurrency))
            groups = {
                name: QuotaPolicy(
                    initial_inflight=concurrency,
                    max_inflight=concurrency,
                    requests_per_minute=policy.requests_per_minute,
                    tokens_per_minute=policy.tokens_per_minute,
                    cooldown_seconds=policy.cooldown_seconds,
                    recovery_successes=policy.recovery_successes,
                    window_seconds=policy.window_seconds,
                ).normalized()
                for name, policy in groups.items()
            }

        configured_global = cfg.get("global_max_inflight", DEFAULT_GLOBAL_MAX_INFLIGHT)
        configured_write = cfg.get("write_batch_size", 100)
        return cls(
            global_max_inflight=max(1, int(global_max_inflight or configured_global)),
            request_timeout_seconds=max(1.0, float(cfg.get("request_timeout_seconds", 60))),
            write_batch_size=max(1, int(write_batch_size or configured_write)),
            write_queue_max_batches=max(1, int(cfg.get("write_queue_max_batches", 8))),
            write_flush_interval_ms=max(1, int(cfg.get("write_flush_interval_ms", 500))),
            existing_id_check_batch_size=max(1, int(cfg.get("existing_id_check_batch_size", 1000))),
            failure_manifest_dir=str(cfg.get("failure_manifest_dir", "./outputs/embedding_failures")),
            retry_policy=retry_policy,
            quota_groups=groups,
            agents=agents,
        )

    def request_batch_size_for(self, agent_name: str) -> int:
        return self.agents[agent_name].request_batch_size


@dataclass(frozen=True)
class EmbeddingBatch:
    agent_name: str
    texts: List[str]
    ids: List[str]
    metadatas: List[Dict[str, Any]]
    estimated_tokens: int


class EmbeddingValidationError(ValueError):
    """Raised when provider output cannot be safely persisted."""


_AUTH_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s,;]+)")
_KEY_RE = re.compile(r"(?i)((?:api[_ -]?key|token)\s*[=:]\s*)([^\s,;]+)")


def sanitize_error_message(message: object, max_length: int = 500) -> str:
    value = str(message or "").replace("\r", " ").replace("\n", " ")
    value = _AUTH_RE.sub(r"\1[REDACTED]", value)
    value = _KEY_RE.sub(r"\1[REDACTED]", value)
    value = " ".join(value.split())
    return value[: max(1, int(max_length))]


@dataclass(frozen=True)
class EmbeddingFailure:
    agent: str
    collection: str
    chunk_id: str
    provider: str
    model: str
    quota_group: str
    attempts: int
    error_type: str
    retryable: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "collection": self.collection,
            "chunk_id": self.chunk_id,
            "provider": self.provider,
            "model": self.model,
            "quota_group": self.quota_group,
            "attempts": max(1, int(self.attempts)),
            "error_type": self.error_type,
            "retryable": bool(self.retryable),
            "message": sanitize_error_message(self.message),
        }


def estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(str(text or "")) / 4))


def build_text_batches(
    agent_name: str,
    texts: Sequence[str],
    ids: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    max_items: int,
    max_tokens: Optional[int] = None,
    token_estimator: Callable[[str], int] = estimate_text_tokens,
) -> List[EmbeddingBatch]:
    if not (len(texts) == len(ids) == len(metadatas)):
        raise ValueError("texts, ids, and metadatas lengths must match")
    item_limit = max(1, int(max_items))
    token_limit = None if max_tokens is None else max(1, int(max_tokens))
    batches: List[EmbeddingBatch] = []
    current_texts: List[str] = []
    current_ids: List[str] = []
    current_metadatas: List[Dict[str, Any]] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_texts, current_ids, current_metadatas, current_tokens
        if not current_texts:
            return
        batches.append(
            EmbeddingBatch(
                agent_name=str(agent_name),
                texts=current_texts,
                ids=current_ids,
                metadatas=current_metadatas,
                estimated_tokens=current_tokens,
            )
        )
        current_texts = []
        current_ids = []
        current_metadatas = []
        current_tokens = 0

    for text, chunk_id, metadata in zip(texts, ids, metadatas):
        value = "" if text is None else str(text)
        if not value.strip():
            raise ValueError(f"blank embedding input for chunk '{chunk_id}'")
        tokens = max(1, int(token_estimator(value)))
        if token_limit is not None and tokens > token_limit:
            raise ValueError(f"embedding input for chunk '{chunk_id}' exceeds max batch tokens")
        if current_texts and (
            len(current_texts) >= item_limit
            or (token_limit is not None and current_tokens + tokens > token_limit)
        ):
            _flush()
        current_texts.append(value)
        current_ids.append(str(chunk_id))
        current_metadatas.append(dict(metadata or {}))
        current_tokens += tokens
    _flush()
    return batches


def validate_embeddings(
    embeddings: Sequence[Sequence[float]],
    expected_count: int,
    expected_dimension: int,
) -> List[List[float]]:
    if len(embeddings) != int(expected_count):
        raise EmbeddingValidationError(
            f"Provider returned {len(embeddings)} vectors for {expected_count} inputs"
        )
    dimension = int(expected_dimension)
    validated: List[List[float]] = []
    for index, vector in enumerate(embeddings):
        if len(vector) != dimension:
            raise EmbeddingValidationError(
                f"Embedding {index} has dimension {len(vector)}; expected {dimension}"
            )
        normalized = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in normalized):
            raise EmbeddingValidationError(f"Embedding {index} contains a non-finite value")
        if not any(value != 0.0 for value in normalized):
            raise EmbeddingValidationError(f"Embedding {index} is an all-zero vector")
        validated.append(normalized)
    return validated


def _status_code(exc: BaseException) -> Optional[int]:
    raw = getattr(exc, "status_code", None)
    if raw is None:
        response = getattr(exc, "response", None)
        raw = getattr(response, "status_code", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def is_throttling_error(exc: BaseException) -> bool:
    status = _status_code(exc)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return status == 429 or "ratelimit" in name or "rate limit" in message or "throttl" in message


def is_retryable_error(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in {408, 409, 429, 500, 502, 503, 504}
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return any(part in name for part in ("timeout", "connection", "temporar", "overload"))


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            now = time.time()
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, target.timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return None


class _QuotaState:
    def __init__(self, policy: QuotaPolicy, clock: Callable[[], float]) -> None:
        self.policy = policy.normalized()
        self.clock = clock
        self.condition = asyncio.Condition()
        self.inflight = 0
        self.peak_inflight = 0
        self.effective_inflight = self.policy.initial_inflight
        self.cooldown_until = 0.0
        self.success_streak = 0
        self.request_events: Deque[float] = deque()
        self.token_events: Deque[tuple[float, int]] = deque()
        self.attempts = 0
        self.successes = 0
        self.retries = 0
        self.throttles = 0
        self.failures = 0
        self.attempt_latencies: Deque[float] = deque(maxlen=10000)

    def _purge(self, now: float) -> None:
        boundary = now - self.policy.window_seconds
        while self.request_events and self.request_events[0] <= boundary:
            self.request_events.popleft()
        while self.token_events and self.token_events[0][0] <= boundary:
            self.token_events.popleft()

    def _rate_wait(self, now: float, tokens: int) -> float:
        waits = [max(0.0, self.cooldown_until - now)]
        rpm = self.policy.requests_per_minute
        if rpm is not None and len(self.request_events) >= rpm:
            waits.append(max(0.0, self.request_events[0] + self.policy.window_seconds - now))
        tpm = self.policy.tokens_per_minute
        used_tokens = sum(value for _, value in self.token_events)
        if tpm is not None and used_tokens + tokens > tpm and self.token_events:
            running = used_tokens
            wait = 0.0
            for timestamp, value in self.token_events:
                running -= value
                wait = max(0.0, timestamp + self.policy.window_seconds - now)
                if running + tokens <= tpm:
                    break
            waits.append(wait)
        return max(waits)

    async def acquire(self, tokens: int) -> None:
        token_count = max(1, int(tokens))
        while True:
            async with self.condition:
                now = self.clock()
                self._purge(now)
                wait_seconds = self._rate_wait(now, token_count)
                if self.inflight < self.effective_inflight and wait_seconds <= 0:
                    self.inflight += 1
                    self.peak_inflight = max(self.peak_inflight, self.inflight)
                    self.attempts += 1
                    self.request_events.append(now)
                    self.token_events.append((now, token_count))
                    return
                if wait_seconds > 0:
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=wait_seconds)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await self.condition.wait()

    async def release(self, success: bool, throttled: bool) -> None:
        async with self.condition:
            self.inflight = max(0, self.inflight - 1)
            if throttled:
                self.throttles += 1
                self.success_streak = 0
                self.effective_inflight = max(1, self.effective_inflight // 2)
                self.cooldown_until = max(
                    self.cooldown_until,
                    self.clock() + self.policy.cooldown_seconds,
                )
            elif success:
                self.successes += 1
                self.success_streak += 1
                if (
                    self.success_streak >= self.policy.recovery_successes
                    and self.effective_inflight < self.policy.max_inflight
                ):
                    self.effective_inflight += 1
                    self.success_streak = 0
            else:
                self.success_streak = 0
            self.condition.notify_all()

    def record_attempt_latency(self, seconds: float) -> None:
        self.attempt_latencies.append(max(0.0, float(seconds)))

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[min(index, len(ordered) - 1)]

    def snapshot(self) -> Dict[str, Any]:
        latencies = list(self.attempt_latencies)
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "inflight": self.inflight,
            "peak_inflight": self.peak_inflight,
            "effective_inflight": self.effective_inflight,
            "max_inflight": self.policy.max_inflight,
            "attempts": self.attempts,
            "successes": self.successes,
            "retries": self.retries,
            "throttles": self.throttles,
            "failures": self.failures,
            "latency_sample_count": len(latencies),
            "average_request_latency_ms": round(average_latency * 1000.0, 3),
            "p50_request_latency_ms": round(self._percentile(latencies, 0.50) * 1000.0, 3),
            "p95_request_latency_ms": round(self._percentile(latencies, 0.95) * 1000.0, 3),
            "p99_request_latency_ms": round(self._percentile(latencies, 0.99) * 1000.0, 3),
        }


class EmbeddingQuotaScheduler:
    """Shared async scheduler for global and quota-group embedding limits."""

    def __init__(
        self,
        global_max_inflight: int,
        quota_groups: Mapping[str, QuotaPolicy],
        retry_policy: Optional[RetryPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.global_max_inflight = max(1, int(global_max_inflight))
        self.global_semaphore = asyncio.Semaphore(self.global_max_inflight)
        self.retry_policy = (retry_policy or RetryPolicy()).normalized()
        self.clock = clock
        self.sleep = sleep
        self.random_value = random_value
        self.groups = {
            str(name): _QuotaState(policy, clock=clock)
            for name, policy in quota_groups.items()
        }
        if not self.groups:
            raise ValueError("At least one embedding quota group is required")

    async def execute(
        self,
        quota_group: str,
        estimated_tokens: int,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        if quota_group not in self.groups:
            raise ValueError(f"Unknown embedding quota group '{quota_group}'")
        state = self.groups[quota_group]
        policy = self.retry_policy

        for attempt in range(1, policy.max_attempts + 1):
            await state.acquire(estimated_tokens)
            try:
                await self.global_semaphore.acquire()
            except BaseException:
                await state.release(success=False, throttled=False)
                raise
            try:
                attempt_started = self.clock()
                result = await operation()
            except BaseException as exc:
                state.record_attempt_latency(self.clock() - attempt_started)
                throttled = is_throttling_error(exc)
                retryable = is_retryable_error(exc)
                await state.release(success=False, throttled=throttled)
                self.global_semaphore.release()
                if not retryable or attempt >= policy.max_attempts:
                    state.failures += 1
                    raise
                state.retries += 1
                delay = None
                if policy.honor_retry_after:
                    delay = retry_after_seconds(exc)
                if delay is None:
                    ceiling = min(
                        policy.max_delay_seconds,
                        policy.base_delay_seconds * (2 ** (attempt - 1)),
                    )
                    delay = max(0.0, ceiling * float(self.random_value()))
                await self.sleep(delay)
                continue
            else:
                state.record_attempt_latency(self.clock() - attempt_started)
                await state.release(success=True, throttled=False)
                self.global_semaphore.release()
                return result

        raise RuntimeError("Embedding retry loop exited unexpectedly")

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: state.snapshot() for name, state in self.groups.items()}


@dataclass(frozen=True)
class EmbeddingWriteItem:
    agent_name: str
    collection_name: str
    provider: str
    model: str
    quota_group: str
    vector_store: Any
    texts: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    ids: List[str]

    def validate(self) -> None:
        if not (
            len(self.texts)
            == len(self.embeddings)
            == len(self.metadatas)
            == len(self.ids)
        ):
            raise ValueError("write item documents, embeddings, metadatas, and ids must match")


@dataclass
class _WriteBuffer:
    agent_name: str
    collection_name: str
    provider: str
    model: str
    quota_group: str
    vector_store: Any
    texts: List[str] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)

    def append(self, item: EmbeddingWriteItem) -> None:
        self.texts.extend(item.texts)
        self.embeddings.extend(item.embeddings)
        self.metadatas.extend(item.metadatas)
        self.ids.extend(item.ids)

    def take(self, count: int) -> tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]]:
        size = min(max(1, int(count)), len(self.ids))
        texts = self.texts[:size]
        embeddings = self.embeddings[:size]
        metadatas = self.metadatas[:size]
        ids = self.ids[:size]
        del self.texts[:size]
        del self.embeddings[:size]
        del self.metadatas[:size]
        del self.ids[:size]
        return texts, embeddings, metadatas, ids


_WRITE_SENTINEL = object()


class EmbeddingWritePipeline:
    """Bounded, single-thread Chroma write pipeline."""

    def __init__(
        self,
        write_batch_size: int,
        queue_max_batches: int = 8,
        flush_interval_ms: int = 500,
    ) -> None:
        self.write_batch_size = max(1, int(write_batch_size))
        self.flush_interval_seconds = max(0.001, float(flush_interval_ms) / 1000.0)
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(1, int(queue_max_batches)))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-writer")
        self.newly_added: Dict[str, int] = {}
        self.failures: List[EmbeddingFailure] = []
        self.peak_queue_depth = 0
        self._task: Optional[asyncio.Task[None]] = None
        self._closed = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="embedding-chroma-writer")

    async def submit(self, item: EmbeddingWriteItem) -> None:
        if self._closed:
            raise RuntimeError("embedding write pipeline is closed")
        item.validate()
        if self._task is None:
            await self.start()
        await self.queue.put(item)
        self.peak_queue_depth = max(self.peak_queue_depth, self.queue.qsize())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._task is None:
                await self.start()
            assert self._task is not None
            if not self._task.done():
                await self.queue.put(_WRITE_SENTINEL)
            await self._task
        finally:
            self.executor.shutdown(wait=True)

    async def _run(self) -> None:
        buffers: Dict[str, _WriteBuffer] = {}
        while True:
            try:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.flush_interval_seconds,
                )
            except asyncio.TimeoutError:
                await self._flush_all(buffers)
                continue

            try:
                if item is _WRITE_SENTINEL:
                    break
                assert isinstance(item, EmbeddingWriteItem)
                key = item.collection_name
                buffer = buffers.get(key)
                if buffer is None:
                    buffer = _WriteBuffer(
                        agent_name=item.agent_name,
                        collection_name=item.collection_name,
                        provider=item.provider,
                        model=item.model,
                        quota_group=item.quota_group,
                        vector_store=item.vector_store,
                    )
                    buffers[key] = buffer
                buffer.append(item)
                while len(buffer.ids) >= self.write_batch_size:
                    await self._flush_buffer(buffer, self.write_batch_size)
            finally:
                self.queue.task_done()

        await self._flush_all(buffers)

    async def _flush_all(self, buffers: Mapping[str, _WriteBuffer]) -> None:
        for buffer in buffers.values():
            while buffer.ids:
                await self._flush_buffer(buffer, self.write_batch_size)

    async def _flush_buffer(self, buffer: _WriteBuffer, count: int) -> None:
        texts, embeddings, metadatas, ids = buffer.take(count)
        if not ids:
            return
        loop = asyncio.get_running_loop()

        def _write() -> None:
            buffer.vector_store.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        try:
            await loop.run_in_executor(self.executor, _write)
        except Exception as exc:
            for chunk_id in ids:
                self.failures.append(
                    EmbeddingFailure(
                        agent=buffer.agent_name,
                        collection=buffer.collection_name,
                        chunk_id=chunk_id,
                        provider=buffer.provider,
                        model=buffer.model,
                        quota_group=buffer.quota_group,
                        attempts=1,
                        error_type="vector_store_write",
                        retryable=True,
                        message=str(exc),
                    )
                )
            return
        self.newly_added[buffer.agent_name] = self.newly_added.get(buffer.agent_name, 0) + len(ids)


def write_failure_manifest(path: Path | str, failures: Sequence[EmbeddingFailure]) -> Optional[Path]:
    if not failures:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for failure in failures:
            handle.write(json.dumps(failure.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, target)
    return target
