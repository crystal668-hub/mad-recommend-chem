"""
======================================================
Batch Vector Database Construction Script
Function: Build 4 Chroma collections (agent1~agent4) using each agent's embedding_model in config
======================================================
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure repo root is importable when running this script from arbitrary cwd.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.environment import load_project_environment

load_project_environment(project_root / ".env")

from agents.agent_config import AgentConfig
from database.embedder import MultiModelEmbedder
from database.embedding_runtime import (
    EmbeddingBatch,
    EmbeddingFailure,
    EmbeddingQuotaScheduler,
    EmbeddingRuntimeSettings,
    EmbeddingWriteItem,
    EmbeddingWritePipeline,
    build_text_batches,
    is_retryable_error,
    is_throttling_error,
    validate_embeddings,
    write_failure_manifest,
)
from database.literature_types import LITERATURE_TYPE_CONFIGS
from database.text_processor import TextProcessor
from database.vector_store import VectorStore
from utils.logger import Logger, setup_logging


_DOI_PREFIX_RE = re.compile(r"(?i)^10\.\d{4,9}/")


def _default_agent_order(llm_cfg: Dict) -> List[str]:
    """Prefer agent1~agent4 ordering; fall back to sorted keys."""
    llm_cfg = llm_cfg or {}
    preferred = ["agent1", "agent2", "agent3", "agent4"]
    configured = list(llm_cfg.keys())
    ordered = [a for a in preferred if a in configured]
    ordered.extend(sorted([a for a in configured if a not in ordered]))
    return ordered


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return int(v)
        except Exception:
            return None
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _prepare_chroma_ids_and_metadatas(
    texts: List[str],
    metadatas: List[Dict],
) -> tuple[List[str], List[Dict]]:
    """
    Precompute stable Chroma ids + metadata for all chunks.

    Why:
    - We build multiple collections (one per agent). We must keep ids consistent across runs.
    - We also may write in multiple `add_documents` calls (streaming). Dedup must work globally,
      not only within a single `add_documents` call.

    Output metadata schema (per chunk):
      - doc_id: str
      - reaction_type: str
      - chunk_index: int (original numeric index)
      - chunk_id: str (Chroma id; e.g. "<doi>#chunk:<idx>" or "hash_<sha256>")
      - total_chunks: int (if provided)
    """
    if len(texts) != len(metadatas):
        raise ValueError("texts and metadatas lengths must match")

    ids: List[str] = []
    out_metas: List[Dict] = []
    seen: set[str] = set()

    for text, meta_in in zip(texts, metadatas):
        meta = dict(meta_in or {})

        doc_id = (meta.get("doc_id") or "").strip()
        chunk_index = _coerce_int(meta.get("chunk_id"))
        if chunk_index is not None:
            meta["chunk_index"] = chunk_index

        if doc_id and _DOI_PREFIX_RE.match(doc_id) and chunk_index is not None:
            chunk_uid = f"{doc_id}#chunk:{chunk_index}"
        else:
            digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
            chunk_uid = f"hash_{digest}"

        # Ensure ids are unique globally. If we hit a collision (rare),
        # disambiguate deterministically using a short hash suffix.
        if chunk_uid in seen:
            payload = {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "reaction_type": meta.get("reaction_type"),
            }
            salt = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
            digest2 = hashlib.sha256(((text or "") + "|" + salt).encode("utf-8")).hexdigest()[:16]
            base_duplicate_uid = f"{chunk_uid}#dup:{digest2}"
            chunk_uid = base_duplicate_uid
            duplicate_index = 2
            while chunk_uid in seen:
                chunk_uid = f"{base_duplicate_uid}:{duplicate_index}"
                duplicate_index += 1

        meta["chunk_id"] = chunk_uid
        ids.append(chunk_uid)
        out_metas.append(meta)
        seen.add(chunk_uid)

    return ids, out_metas


@dataclass
class _AgentEmbeddingContext:
    agent_name: str
    collection_name: str
    provider: str
    model: str
    dimension: int
    quota_group: str
    embedder: Any
    vector_store: Any
    batches: Sequence[EmbeddingBatch]
    already_present: int = 0


@dataclass
class _EmbeddingPipelineOutcome:
    agent_results: Dict[str, Dict[str, object]]
    failures: List[EmbeddingFailure]
    scheduler_snapshot: Dict[str, Dict[str, Any]]
    peak_write_queue_depth: int
    wall_clock_seconds: float


@dataclass
class _AgentRequestMetrics:
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    latencies_seconds: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    def record(self, started_at: float, finished_at: float, batch_size: int) -> None:
        if self.started_at is None or started_at < self.started_at:
            self.started_at = started_at
        if self.finished_at is None or finished_at > self.finished_at:
            self.finished_at = finished_at
        self.latencies_seconds.append(max(0.0, finished_at - started_at))
        self.batch_sizes.append(max(0, int(batch_size)))

    def percentile_ms(self, percentile: float) -> float:
        if not self.latencies_seconds:
            return 0.0
        ordered = sorted(self.latencies_seconds)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[min(index, len(ordered) - 1)] * 1000.0, 3)

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)


class _FailFastStop(RuntimeError):
    """Internal signal used to stop producers after recording a batch failure."""


def _get_missing_indices(
    vector_store: Any,
    ids: Sequence[str],
    batch_size: int,
) -> tuple[List[int], int]:
    existing: set[str] = set()
    size = max(1, int(batch_size))
    for start in range(0, len(ids), size):
        candidate_ids = list(ids[start:start + size])
        existing.update(vector_store.get_existing_ids(candidate_ids))
    missing = [index for index, chunk_id in enumerate(ids) if chunk_id not in existing]
    return missing, len(existing)


async def _run_embedding_pipeline(
    contexts: Sequence[_AgentEmbeddingContext],
    settings: EmbeddingRuntimeSettings,
    continue_on_error: bool = True,
) -> _EmbeddingPipelineOutcome:
    pipeline_started = time.monotonic()
    scheduler = EmbeddingQuotaScheduler(
        global_max_inflight=settings.global_max_inflight,
        quota_groups=settings.quota_groups,
        retry_policy=settings.retry_policy,
    )
    writer = EmbeddingWritePipeline(
        write_batch_size=settings.write_batch_size,
        queue_max_batches=settings.write_queue_max_batches,
        flush_interval_ms=settings.write_flush_interval_ms,
    )
    failures: List[EmbeddingFailure] = []
    request_metrics = {
        context.agent_name: _AgentRequestMetrics()
        for context in contexts
    }
    await writer.start()

    async def _process_batch(context: _AgentEmbeddingContext, request: EmbeddingBatch) -> None:
        request_started = time.monotonic()
        async def _network_attempt() -> List[List[float]]:
            return await asyncio.wait_for(
                context.embedder.embed_documents_batch(
                    list(request.texts),
                    agent_name=context.agent_name,
                ),
                timeout=settings.request_timeout_seconds,
            )

        try:
            raw_embeddings = await scheduler.execute(
                context.quota_group,
                request.estimated_tokens,
                _network_attempt,
            )
            embeddings = validate_embeddings(
                raw_embeddings,
                expected_count=len(request.texts),
                expected_dimension=context.dimension,
            )
            await writer.submit(
                EmbeddingWriteItem(
                    agent_name=context.agent_name,
                    collection_name=context.collection_name,
                    provider=context.provider,
                    model=context.model,
                    quota_group=context.quota_group,
                    vector_store=context.vector_store,
                    texts=list(request.texts),
                    embeddings=embeddings,
                    metadatas=[dict(metadata) for metadata in request.metadatas],
                    ids=list(request.ids),
                )
            )
        except Exception as exc:
            retryable = is_retryable_error(exc)
            error_type = "rate_limit" if is_throttling_error(exc) else type(exc).__name__
            attempts = settings.retry_policy.max_attempts if retryable else 1
            for chunk_id in request.ids:
                failures.append(
                    EmbeddingFailure(
                        agent=context.agent_name,
                        collection=context.collection_name,
                        chunk_id=chunk_id,
                        provider=context.provider,
                        model=context.model,
                        quota_group=context.quota_group,
                        attempts=attempts,
                        error_type=error_type,
                        retryable=retryable,
                        message=str(exc),
                    )
                )
            if not continue_on_error:
                raise _FailFastStop() from exc
        finally:
            request_metrics[context.agent_name].record(
                request_started,
                time.monotonic(),
                len(request.ids),
            )

    async def _run_agent(context: _AgentEmbeddingContext) -> None:
        window_size = settings.quota_groups[context.quota_group].max_inflight
        iterator = iter(context.batches)
        pending: set[asyncio.Task[None]] = set()

        def _fill_window() -> None:
            while len(pending) < window_size:
                try:
                    request = next(iterator)
                except StopIteration:
                    return
                pending.add(
                    asyncio.create_task(
                        _process_batch(context, request),
                        name=f"embed-{context.agent_name}",
                    )
                )

        try:
            _fill_window()
            while pending:
                completed, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                completed_results = await asyncio.gather(
                    *completed,
                    return_exceptions=True,
                )
                for result in completed_results:
                    if isinstance(result, BaseException):
                        raise result
                _fill_window()
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    producer_tasks = [
        asyncio.create_task(_run_agent(context), name=f"producer-{context.agent_name}")
        for context in contexts
    ]
    producer_error: Optional[BaseException] = None
    try:
        await asyncio.gather(*producer_tasks)
    except BaseException as exc:
        producer_error = exc
        for task in producer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)
    finally:
        try:
            await writer.close()
        finally:
            close_tasks = []
            for context in contexts:
                close_fn = getattr(context.embedder, "aclose", None)
                if callable(close_fn):
                    close_tasks.append(close_fn())
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

    failures.extend(writer.failures)
    if producer_error is not None and not isinstance(producer_error, _FailFastStop):
        raise producer_error

    agent_results: Dict[str, Dict[str, object]] = {}
    for context in contexts:
        agent_failures = sum(1 for failure in failures if failure.agent == context.agent_name)
        newly_added = writer.newly_added.get(context.agent_name, 0)
        successful = context.already_present + newly_added
        if agent_failures:
            status = "partial" if successful else "error"
        else:
            status = "ok"
        metrics = request_metrics[context.agent_name]
        planned_chunks = sum(len(request.ids) for request in context.batches)
        planned_requests = len(context.batches)
        request_batch_limit = settings.agents[context.agent_name].request_batch_size
        batch_fill_ratio = (
            planned_chunks / (planned_requests * request_batch_limit)
            if planned_requests and request_batch_limit
            else 0.0
        )
        request_reduction_ratio = (
            1.0 - (planned_requests / planned_chunks)
            if planned_chunks
            else 0.0
        )
        active_seconds = metrics.duration_seconds
        agent_results[context.agent_name] = {
            "status": status,
            "collection_name": context.collection_name,
            "document_count": context.vector_store.get_collection_count(),
            "embedding_model": context.model,
            "embedding_provider": context.provider,
            "quota_group": context.quota_group,
            "request_batch_size": request_batch_limit,
            "already_present": context.already_present,
            "newly_added": newly_added,
            "failed": agent_failures,
            "planned_chunks": planned_chunks,
            "logical_request_count": planned_requests,
            "completed_request_count": len(metrics.latencies_seconds),
            "average_request_batch_size": round(
                planned_chunks / planned_requests if planned_requests else 0.0,
                3,
            ),
            "request_batch_fill_ratio": round(batch_fill_ratio, 6),
            "request_reduction_vs_scalar_ratio": round(request_reduction_ratio, 6),
            "active_embedding_seconds": round(active_seconds, 3),
            "embedding_chunks_per_second": round(
                newly_added / active_seconds if active_seconds else 0.0,
                3,
            ),
            "p50_logical_request_latency_ms": metrics.percentile_ms(0.50),
            "p95_logical_request_latency_ms": metrics.percentile_ms(0.95),
            "p99_logical_request_latency_ms": metrics.percentile_ms(0.99),
        }

    return _EmbeddingPipelineOutcome(
        agent_results=agent_results,
        failures=failures,
        scheduler_snapshot=scheduler.snapshot(),
        peak_write_queue_depth=writer.peak_queue_depth,
        wall_clock_seconds=round(time.monotonic() - pipeline_started, 3),
    )


def build_vector_databases_batch(
    config_path: str = "./config/config.yaml",
    data_dir: str = "./data/raw",
    literature_type_configs: Optional[Dict[str, Dict]] = None,
    agent_names: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    embedding_request_batch_size: Optional[int] = None,
    embedding_concurrency: Optional[int] = None,
    max_workers: int = 4,
    sleep_between_batches: Optional[float] = None,
    embedding_write_batch_size: Optional[int] = None,
    embedding_global_max_inflight: Optional[int] = None,
    max_chunks: Optional[int] = None,
    persist_directory_override: Optional[str] = None,
    base_collection_name_override: Optional[str] = None,
    report_path: Optional[str] = None,
    resume: bool = True,
    clear_existing: Optional[bool] = None,
    skip_if_exists: bool = False,
    continue_on_error: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Batch build vector databases for multiple agents.

    It loads and chunks documents once, then for each agent:
      - uses that agent's embedding_model from config to compute embeddings
      - writes to a dedicated Chroma collection: <base_collection_name>_<agent_name>

    Args:
        config_path: 配置文件路径
        data_dir: 原始数据目录
        literature_type_configs: Literature Type directory and CSV metadata configuration
        agent_names: 要构建的agent列表（None则使用config里的agent1~agent4顺序）
        chunk_size: 分块大小（默认使用config.rag.chunk_size；CLI可显式覆盖）
        chunk_overlap: 分块重叠（默认使用config.rag.chunk_overlap；CLI可显式覆盖）
        embedding_batch_size: 兼容参数；覆盖所有agent的provider request batch大小
        embedding_request_batch_size: 覆盖所有agent的provider request batch大小
        embedding_concurrency: 兼容参数；覆盖所有quota group的并发上限
        max_workers: 兼容参数；新async scheduler不再使用agent线程池
        sleep_between_batches: 已弃用；新scheduler使用quota窗口和Retry-After
        embedding_write_batch_size: Chroma writer批大小覆盖值
        embedding_global_max_inflight: 进程级embedding在途请求覆盖值
        resume: 断点续跑（默认True）；为True时会跳过已存在的chunk并补齐缺失部分
        clear_existing: True=自动清空已有collection；False=不清空（若已有则按skip_if_exists策略处理）；None=交互式询问
        skip_if_exists: collection已有数据时，跳过该collection的构建（避免重复id导致Chroma报错）
        continue_on_error: 某个agent失败后是否继续构建其他agent

    Returns:
        Dict[str, Dict]: per-agent result summary.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    config = AgentConfig(config_path)
    setup_logging(config.config, run_id=f"build_vector_db_batch_{timestamp}")
    logger = Logger.get_logger("MAD.build_vector_db_batch")

    logger.info("Starting batch Chroma vector database build", extra={"event": "vector_db.batch_build.start"})

    if literature_type_configs is None:
        literature_type_configs = LITERATURE_TYPE_CONFIGS

    # -------------------------
    # 1) Load config
    # -------------------------
    logger.info("[Step 1/5] Loading configuration...")
    vector_config = config.get_vector_store_config()
    rag_config = config.get_rag_config()

    cfg_chunk_size = rag_config.get("chunk_size")
    cfg_chunk_overlap = rag_config.get("chunk_overlap")
    if chunk_size is None:
        chunk_size = int(cfg_chunk_size) if cfg_chunk_size is not None else 256
    else:
        chunk_size = int(chunk_size)

    if chunk_overlap is None:
        chunk_overlap = int(cfg_chunk_overlap) if cfg_chunk_overlap is not None else 50
    else:
        chunk_overlap = int(chunk_overlap)

    llm_cfg_root = (config.config or {}).get("llm", {}) or {}
    if agent_names is None:
        agent_names = _default_agent_order(llm_cfg_root)

    # Keep only configured agents (avoid hard failure when user passes an unknown name).
    agent_names = [a for a in agent_names if a in llm_cfg_root]
    if not agent_names:
        raise ValueError("No valid agent names found in config.llm")

    base_collection_name = (
        base_collection_name_override
        or vector_config.get("collection_name", "chemical_reactions_recommendation")
    )
    persist_directory = (
        persist_directory_override
        or vector_config.get("persist_directory", "./data/chroma_db")
    )
    distance_metric = vector_config.get("distance_metric", "cosine")

    logger.info(f"✓ Agents: {agent_names}")
    logger.info(f"✓ persist_directory: {persist_directory}")
    logger.info(f"✓ base_collection_name: {base_collection_name}")
    logger.info(f"✓ chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
    logger.info(f"✓ legacy max_workers: {max_workers}")
    logger.info(f"✓ literature_types: {list(literature_type_configs)}")

    # Build all agent configs (used by MultiModelEmbedder to select per-agent embedding provider/model).
    all_agent_configs = {name: config.get_llm_config(name) for name in agent_names}

    # -------------------------
    # 2) Load & chunk docs (once)
    # -------------------------
    logger.info("\n[Step 2/5] Loading literature data...")
    processor = TextProcessor(data_dir)

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"\n✗ Data directory does not exist: {data_dir}")
        logger.error("  Please ensure each Literature Type has Markdown and CSV metadata:")
        for _, cfg in literature_type_configs.items():
            logger.error(f"    {data_dir}/{cfg['path']}/*.md")
            logger.error(f"    {cfg['metadata_csv']}")
        return {}

    documents = processor.load_literature_type_documents(
        base_dir=data_dir,
        literature_type_configs=literature_type_configs,
    )
    logger.info(f"\n✓ Loaded {len(documents)} Document objects")
    if not documents:
        logger.error("\n✗ No documents found, please check the data directory (supported: .md)")
        return {}

    logger.info("\n[Step 3/5] Chunking documents...")
    chunked_documents = processor.chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info(f"✓ Number of chunked documents: {len(chunked_documents)}")
    if not chunked_documents:
        logger.error("\n✗ No chunks produced, cannot build vector database")
        return {}
    if max_chunks is not None:
        chunk_limit = int(max_chunks)
        if chunk_limit < 1:
            raise ValueError("max_chunks must be at least 1")
        if len(chunked_documents) > chunk_limit:
            chunked_documents = chunked_documents[:chunk_limit]
            logger.info(f"Load-test chunk limit applied: {len(chunked_documents)}")

    texts = [doc.text for doc in chunked_documents]
    total_chunks = len(texts)
    raw_metadatas = [dict(doc.metadata or {}) for doc in chunked_documents]
    # Precompute stable ids + prepared metadata (chunk_id becomes a string uid; chunk_index preserved).
    chunk_ids, base_metadatas = _prepare_chroma_ids_and_metadatas(texts, raw_metadatas)

    # -------------------------
    # 4) Init embedder (once)
    # -------------------------
    # Note: for concurrency we instantiate one embedder per agent (thread-safety + per-provider clients).
    logger.info("\n[Step 4/5] Preparing embedders...")

    # -------------------------
    # 5) Build collections per agent
    # -------------------------
    logger.info("\n[Step 5/5] Building collections...")

    results: Dict[str, Dict[str, object]] = {}
    # Preflight: create/reset collections sequentially (avoid interactive prompt races + reduce Chroma lock contention).
    build_plan: Dict[str, Dict[str, object]] = {}
    vector_stores: Dict[str, VectorStore] = {}

    for agent_name in agent_names:
        agent_cfg = all_agent_configs.get(agent_name, {}) or {}
        embedding_model = agent_cfg.get("embedding_model")
        embedding_provider = agent_cfg.get("embedding_provider")
        llm_model = agent_cfg.get("model")

        collection_name = f"{base_collection_name}_{agent_name}"
        logger.info("=" * 60)
        logger.info(
            f"Preflight collection: {collection_name} | agent={agent_name} | llm_model={llm_model} | "
            f"embedding_model={embedding_model} | provider={embedding_provider}"
        )

        try:
            vector_store = VectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=None,  # precomputed embeddings
                distance_metric=distance_metric,
            )

            current_count = vector_store.get_collection_count()
            if current_count > 0:
                if skip_if_exists:
                    logger.info(f"Collection already has {current_count} documents; skip_if_exists=True so skipping.")
                    results[agent_name] = {
                        "status": "skipped",
                        "collection_name": collection_name,
                        "document_count": current_count,
                        "embedding_model": embedding_model,
                        "embedding_provider": embedding_provider,
                    }
                    continue

                if clear_existing is True:
                    vector_store.reset_collection()
                    logger.info("✓ Collection cleared (clear_existing=True)")
                elif clear_existing is False:
                    if resume:
                        logger.info(
                            f"Collection already has {current_count} documents; clear_existing=False but resume=True so will "
                            "check chunk ids and only embed/add missing."
                        )
                    else:
                        raise RuntimeError(
                            f"Collection '{collection_name}' already has {current_count} documents. "
                            f"Refusing to add duplicates (enable --resume, or use --clear / --skip-if-exists)."
                        )
                else:
                    if resume:
                        # Default: resumable indexing (skip existing chunk ids and only add missing).
                        logger.info(
                            f"Resume enabled: collection has {current_count} docs; will check chunk ids and only embed/add missing."
                        )
                    else:
                        # Legacy behaviour: ask whether to clear; if not, abort this agent.
                        prompt = f"\nCollection '{collection_name}' already has {current_count} documents. Clear it? (y/n): "
                        logger.info(prompt)
                        user_input = input(prompt)
                        if user_input.lower() == "y":
                            vector_store.reset_collection()
                            logger.info("✓ Collection cleared")
                        else:
                            raise RuntimeError(
                                f"User chose not to clear collection '{collection_name}'. "
                                f"Aborting this agent (use --skip-if-exists or --resume to continue)."
                            )

            build_plan[agent_name] = {
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            }
            vector_stores[agent_name] = vector_store

        except Exception as e:
            logger.error(f"✗ Preflight failed for {agent_name}: {str(e)}", exc_info=True)
            results[agent_name] = {
                "status": "error",
                "collection_name": collection_name,
                "error": str(e),
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            }
            if not continue_on_error:
                return results

    if not build_plan:
        logger.info("No collections to build (all skipped or errored).")
        return results

    runtime_config = (config.config or {}).get("embedding_runtime", {}) or {}
    settings = EmbeddingRuntimeSettings.from_config(
        runtime_config,
        agent_configs=all_agent_configs,
        legacy_batch_size=(
            embedding_request_batch_size
            if embedding_request_batch_size is not None
            else embedding_batch_size
        ),
        legacy_concurrency=embedding_concurrency,
        write_batch_size=embedding_write_batch_size,
        global_max_inflight=embedding_global_max_inflight,
    )
    if embedding_batch_size is not None:
        logger.warning("--embedding-batch-size is deprecated; use per-agent embedding_request_batch_size")
    if embedding_concurrency is not None:
        logger.warning("--embedding-concurrency is deprecated; use embedding_runtime.quota_groups")
    if sleep_between_batches not in (None, 0, 0.0):
        logger.warning("sleep_between_batches is deprecated and ignored by the quota scheduler")
    if max_workers != 4:
        logger.warning("max_workers is deprecated and ignored by the async embedding scheduler")

    logger.info(
        f"Embedding runtime: global_max_inflight={settings.global_max_inflight}, "
        f"write_batch_size={settings.write_batch_size}, quota_groups={list(settings.quota_groups)}"
    )

    contexts: List[_AgentEmbeddingContext] = []
    for agent_name in build_plan:
        agent_cfg = all_agent_configs.get(agent_name, {}) or {}
        embedder = MultiModelEmbedder(agent_cfg, agent_configs=all_agent_configs)
        vector_store = vector_stores[agent_name]
        profile = embedder.agent_embedding_profiles.get(agent_name, {}) or {}
        model_name = embedder.get_model_for_agent(agent_name)
        dimension = embedder.get_embedding_dimension(model_name)
        runtime_agent = settings.agents[agent_name]

        if resume:
            missing_indices, already_present = _get_missing_indices(
                vector_store,
                chunk_ids,
                settings.existing_id_check_batch_size,
            )
        else:
            missing_indices = list(range(len(chunk_ids)))
            already_present = 0

        missing_texts = [texts[index] for index in missing_indices]
        missing_ids = [chunk_ids[index] for index in missing_indices]
        missing_metadatas = [base_metadatas[index] for index in missing_indices]
        request_batches = build_text_batches(
            agent_name=agent_name,
            texts=missing_texts,
            ids=missing_ids,
            metadatas=missing_metadatas,
            max_items=runtime_agent.request_batch_size,
            max_tokens=runtime_agent.max_batch_tokens,
        )
        logger.info(
            f"[{agent_name}] Prepared embedding pipeline: missing={len(missing_ids)}, "
            f"already_present={already_present}, request_batches={len(request_batches)}, "
            f"batch_size={runtime_agent.request_batch_size}, quota_group={runtime_agent.quota_group}"
        )
        contexts.append(
            _AgentEmbeddingContext(
                agent_name=agent_name,
                collection_name=str(build_plan[agent_name]["collection_name"]),
                provider=str(profile.get("embedding_provider") or agent_cfg.get("embedding_provider") or ""),
                model=model_name,
                dimension=dimension,
                quota_group=runtime_agent.quota_group,
                embedder=embedder,
                vector_store=vector_store,
                batches=request_batches,
                already_present=already_present,
            )
        )

    outcome = asyncio.run(
        _run_embedding_pipeline(
            contexts,
            settings,
            continue_on_error=continue_on_error,
        )
    )
    results.update(outcome.agent_results)
    manifest_path: Optional[Path] = None
    if outcome.failures:
        manifest_path = Path(settings.failure_manifest_dir) / f"build_vector_db_batch_{timestamp}.jsonl"
        write_failure_manifest(manifest_path, outcome.failures)
        for agent_result in outcome.agent_results.values():
            if int(agent_result.get("failed", 0)) > 0:
                agent_result["failure_manifest"] = str(manifest_path)

    logger.info(
        "Embedding scheduler summary",
        extra={
            "event": "embedding.scheduler.summary",
            "quota_groups": outcome.scheduler_snapshot,
            "peak_write_queue_depth": outcome.peak_write_queue_depth,
            "wall_clock_seconds": outcome.wall_clock_seconds,
            "failure_count": len(outcome.failures),
            "failure_manifest": str(manifest_path) if manifest_path else None,
        },
    )
    if report_path:
        report_target = Path(report_path)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "run_id": f"build_vector_db_batch_{timestamp}",
            "sample_chunks": total_chunks,
            "agents": outcome.agent_results,
            "runtime": {
                "wall_clock_seconds": outcome.wall_clock_seconds,
                "global_max_inflight": settings.global_max_inflight,
                "write_batch_size": settings.write_batch_size,
                "peak_write_queue_depth": outcome.peak_write_queue_depth,
                "quota_groups": outcome.scheduler_snapshot,
                "failure_count": len(outcome.failures),
                "failure_manifest": str(manifest_path) if manifest_path else None,
                "persist_directory": str(persist_directory),
                "base_collection_name": str(base_collection_name),
            },
        }
        temporary_report = report_target.with_suffix(report_target.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_report.replace(report_target)
        logger.info(f"Embedding load report written to {report_target}")
    for agent_name, agent_result in outcome.agent_results.items():
        logger.info(
            f"[{agent_name}] Done: status={agent_result['status']}, "
            f"count={agent_result['document_count']}, already_present={agent_result['already_present']}, "
            f"newly_added={agent_result['newly_added']}, failed={agent_result['failed']}"
        )

    logger.info("=" * 60)
    logger.info("Batch build complete")
    logger.info("=" * 60)
    return results


def _parse_agent_list(value: str) -> List[str]:
    if not value:
        return []
    # Allow "agent1,agent2" or "agent1 agent2"
    raw = value.replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch build Chroma collections for agent1~agent4.")
    parser.add_argument("--config", dest="config_path", default="./config/config.yaml", help="config yaml path")
    parser.add_argument("--data-dir", dest="data_dir", default="./data/raw", help="raw markdown data dir")
    parser.add_argument(
        "--agents",
        dest="agents",
        default="agent1,agent2,agent3,agent4",
        help="comma/space separated agent list (default: agent1,agent2,agent3,agent4)",
    )
    parser.add_argument(
        "--literature-types",
        dest="literature_types",
        default=None,
        help="optional comma-separated literature types to load",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=None,
        help="chunk size (default: config.rag.chunk_size or 256)",
    )
    parser.add_argument(
        "--chunk-overlap",
        dest="chunk_overlap",
        type=int,
        default=None,
        help="chunk overlap (default: config.rag.chunk_overlap or 50)",
    )
    parser.add_argument(
        "--embedding-request-batch-size",
        dest="embedding_request_batch_size",
        type=int,
        default=None,
        help="override provider request batch size",
    )
    parser.add_argument(
        "--embedding-batch-size",
        dest="embedding_batch_size",
        type=int,
        default=None,
        help="deprecated alias for --embedding-request-batch-size",
    )
    parser.add_argument(
        "--embedding-concurrency",
        dest="embedding_concurrency",
        type=int,
        default=None,
        help="deprecated override for every quota-group concurrency limit",
    )
    parser.add_argument(
        "--embedding-write-batch-size",
        dest="embedding_write_batch_size",
        type=int,
        default=None,
        help="override Chroma writer batch size",
    )
    parser.add_argument(
        "--embedding-global-max-inflight",
        dest="embedding_global_max_inflight",
        type=int,
        default=None,
        help="override process-wide embedding request concurrency",
    )
    parser.add_argument(
        "--max-chunks",
        dest="max_chunks",
        type=int,
        default=None,
        help="limit the shared input sample; intended for controlled load tests",
    )
    parser.add_argument(
        "--persist-directory",
        dest="persist_directory_override",
        default=None,
        help="override Chroma persistence directory",
    )
    parser.add_argument(
        "--collection-name",
        dest="base_collection_name_override",
        default=None,
        help="override base collection name",
    )
    parser.add_argument(
        "--report-path",
        dest="report_path",
        default=None,
        help="write a sanitized JSON load report",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="resume from existing collection data")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="disable resume; require clear/skip")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=4,
        help="deprecated; async scheduler replaces per-agent worker threads",
    )
    parser.add_argument(
        "--sleep-between-batches",
        dest="sleep_between_batches",
        type=float,
        default=None,
        help="deprecated; quota scheduler controls request pacing",
    )
    parser.add_argument(
        "--clear",
        dest="clear_existing",
        action="store_true",
        help="clear existing collections without prompting",
    )
    parser.add_argument(
        "--skip-if-exists",
        dest="skip_if_exists",
        action="store_true",
        help="skip collections that already have documents",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help="stop when any agent build fails",
    )
    args = parser.parse_args()

    agent_names = _parse_agent_list(args.agents)
    selected_literature_types = LITERATURE_TYPE_CONFIGS
    if args.literature_types:
        requested_types = [
            name.strip()
            for name in args.literature_types.split(",")
            if name.strip()
        ]
        unknown_types = [name for name in requested_types if name not in LITERATURE_TYPE_CONFIGS]
        if unknown_types:
            parser.error(f"unknown literature types: {', '.join(unknown_types)}")
        selected_literature_types = {
            name: LITERATURE_TYPE_CONFIGS[name]
            for name in requested_types
        }
    # When user explicitly passes --clear, prefer non-interactive clearing.
    clear_existing: Optional[bool] = True if args.clear_existing else None

    build_vector_databases_batch(
        config_path=args.config_path,
        data_dir=args.data_dir,
        literature_type_configs=selected_literature_types,
        agent_names=agent_names,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        embedding_request_batch_size=args.embedding_request_batch_size,
        embedding_concurrency=args.embedding_concurrency,
        max_workers=args.max_workers,
        sleep_between_batches=args.sleep_between_batches,
        embedding_write_batch_size=args.embedding_write_batch_size,
        embedding_global_max_inflight=args.embedding_global_max_inflight,
        max_chunks=args.max_chunks,
        persist_directory_override=args.persist_directory_override,
        base_collection_name_override=args.base_collection_name_override,
        report_path=args.report_path,
        resume=bool(args.resume),
        clear_existing=clear_existing,
        skip_if_exists=bool(args.skip_if_exists),
        continue_on_error=not bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
