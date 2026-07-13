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
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Ensure repo root is importable when running this script from arbitrary cwd.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.agent_config import AgentConfig
from database.embedder import MultiModelEmbedder
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
            chunk_uid = f"{chunk_uid}#dup:{digest2}"

        meta["chunk_id"] = chunk_uid
        ids.append(chunk_uid)
        out_metas.append(meta)
        seen.add(chunk_uid)

    return ids, out_metas


def build_vector_databases_batch(
    config_path: str = "./config/config.yaml",
    data_dir: str = "./data/raw",
    literature_type_configs: Optional[Dict[str, Dict]] = None,
    agent_names: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_batch_size: int = 10,
    embedding_concurrency: int = 1,
    max_workers: int = 4,
    sleep_between_batches: float = 0.5,
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
        embedding_batch_size: embedding批大小
        embedding_concurrency: OpenRouter embedding异步并发的batch数量（默认1=关闭；>1启用）
        max_workers: 并发worker数量（默认4；设为1可退回串行）
        sleep_between_batches: 每个agent在embedding batch之间的sleep（秒），用于简单限速（默认0.5）
        resume: 断点续跑（默认True）；为True时会跳过已存在的chunk并补齐缺失部分
        clear_existing: True=自动清空已有collection；False=不清空（若已有则按skip_if_exists策略处理）；None=交互式询问
        skip_if_exists: collection已有数据时，跳过该collection的构建（避免重复id导致Chroma报错）
        continue_on_error: 某个agent失败后是否继续构建其他agent

    Returns:
        Dict[str, Dict]: per-agent result summary.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    base_collection_name = vector_config.get("collection_name", "chemical_reactions_recommendation")
    persist_directory = vector_config.get("persist_directory", "./data/chroma_db")
    distance_metric = vector_config.get("distance_metric", "cosine")

    logger.info(f"✓ Agents: {agent_names}")
    logger.info(f"✓ persist_directory: {persist_directory}")
    logger.info(f"✓ base_collection_name: {base_collection_name}")
    logger.info(f"✓ chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
    logger.info(f"✓ max_workers: {max_workers}")
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

    chroma_write_lock = threading.Lock()

    def _build_one_agent(agent_name: str) -> Dict[str, object]:
        agent_cfg = all_agent_configs.get(agent_name, {}) or {}
        embedding_model = agent_cfg.get("embedding_model")
        embedding_provider = agent_cfg.get("embedding_provider")
        collection_name = str(build_plan[agent_name]["collection_name"])

        # Per-agent embedder (avoid shared client dicts across threads).
        embedder = MultiModelEmbedder(agent_cfg, agent_configs=all_agent_configs)
        vector_store = vector_stores[agent_name]

        total = len(texts)
        model_name = embedder.get_model_for_agent(agent_name)
        dim = embedder.get_embedding_dimension(model_name)
        resolved_provider = (embedder.agent_embedding_profiles.get(agent_name, {}) or {}).get("embedding_provider", "")

        logger.info(
            f"[{agent_name}] Start embedding: total_chunks={total}, model={embedding_model}, provider={embedding_provider}, dim={dim}"
        )

        already_present = 0
        newly_added = 0

        async_enabled = str(resolved_provider or "").lower() == "openrouter" and int(embedding_concurrency) > 1
        pending: List[Dict[str, object]] = []
        loop: Optional[asyncio.AbstractEventLoop] = None
        if async_enabled:
            # Keep a single event loop per agent to avoid cross-loop issues with async HTTP clients.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _embed_many(text_batches: List[List[str]]) -> List[object]:
            tasks = [embedder.embed_texts_openrouter_async(b, agent_name=agent_name) for b in text_batches]
            return await asyncio.gather(*tasks, return_exceptions=True)

        def _flush_pending() -> None:
            nonlocal newly_added
            if not pending:
                return

            text_batches = [p["batch_texts"] for p in pending]  # type: ignore[index]
            if loop is None:
                raise RuntimeError("Async embedding loop is not initialized")
            results = loop.run_until_complete(_embed_many(text_batches))

            for p, r in zip(pending, results):
                batch_texts = p["batch_texts"]  # type: ignore[assignment]
                batch_ids = p["batch_ids"]  # type: ignore[assignment]
                batch_metas = p["batch_metas"]  # type: ignore[assignment]
                bend = int(p["end"])  # type: ignore[arg-type]

                if isinstance(r, Exception):
                    logger.error(f"[{agent_name}] Async batch embedding failed (end={bend}): {str(r)}")
                    batch_embeddings = [[0.0] * dim for _ in batch_texts]  # type: ignore[arg-type]
                else:
                    batch_embeddings = r  # type: ignore[assignment]

                # Chroma persistent backend isn't guaranteed to be safe for concurrent writers.
                # Serialize writes to avoid sqlite "database is locked" issues.
                with chroma_write_lock:
                    vector_store.add_documents(
                        documents=batch_texts,  # type: ignore[arg-type]
                        embeddings=batch_embeddings,  # type: ignore[arg-type]
                        metadatas=[m.copy() for m in batch_metas],  # type: ignore[arg-type]
                        ids=batch_ids,  # type: ignore[arg-type]
                    )
                newly_added += len(batch_ids)  # type: ignore[arg-type]

                if sleep_between_batches and bend < total:
                    time.sleep(float(sleep_between_batches))

                if bend % max(1, int(embedding_batch_size) * 50) == 0 or bend == total:
                    logger.info(f"[{agent_name}] Progress: {bend}/{total}")

            pending.clear()

        # Stream embeddings -> Chroma in small batches to keep memory bounded.
        for start in range(0, total, int(embedding_batch_size)):
            end = min(start + int(embedding_batch_size), total)
            batch_texts = texts[start:end]
            batch_ids = chunk_ids[start:end]
            batch_metas = base_metadatas[start:end]

            # Resume: skip ids that already exist (avoid duplicate-id errors and wasted embedding calls).
            if resume:
                with chroma_write_lock:
                    existing_ids = vector_store.get_existing_ids(batch_ids)
                if existing_ids:
                    already_present += len(existing_ids)

                missing_indices = [i for i, cid in enumerate(batch_ids) if cid not in existing_ids]
                if not missing_indices:
                    continue

                batch_texts = [batch_texts[i] for i in missing_indices]
                batch_ids = [batch_ids[i] for i in missing_indices]
                batch_metas = [batch_metas[i] for i in missing_indices]

            if async_enabled:
                pending.append(
                    {
                        "batch_texts": batch_texts,
                        "batch_ids": batch_ids,
                        "batch_metas": batch_metas,
                        "end": end,
                    }
                )
                if len(pending) >= int(embedding_concurrency) or end == total:
                    _flush_pending()
                continue

            batch_embeddings: List[List[float]] = []
            for text in batch_texts:
                try:
                    batch_embeddings.append(embedder.embed_text(text, agent_name=agent_name))
                except Exception as e:
                    logger.error(f"[{agent_name}] Embedding failed (idx={start + len(batch_embeddings)}): {str(e)}")
                    batch_embeddings.append([0.0] * dim)

            # Chroma persistent backend isn't guaranteed to be safe for concurrent writers.
            # Serialize writes to avoid sqlite "database is locked" issues.
            with chroma_write_lock:
                vector_store.add_documents(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=[m.copy() for m in batch_metas],
                    ids=batch_ids,
                )
            newly_added += len(batch_ids)

            if sleep_between_batches and end < total:
                time.sleep(float(sleep_between_batches))

            if end % max(1, int(embedding_batch_size) * 50) == 0 or end == total:
                logger.info(f"[{agent_name}] Progress: {end}/{total}")

        # Best-effort flush in case loop exits with pending batches.
        if pending:
            _flush_pending()

        if loop is not None:
            try:
                # Best-effort close async clients (API depends on openai version).
                close_awaitables = []
                for c in getattr(embedder, "async_openai_clients", {}).values():
                    close_fn = getattr(c, "close", None)
                    if callable(close_fn):
                        try:
                            maybe_awaitable = close_fn()
                            if asyncio.iscoroutine(maybe_awaitable):
                                close_awaitables.append(maybe_awaitable)
                        except Exception:
                            pass
                if close_awaitables:
                    loop.run_until_complete(asyncio.gather(*close_awaitables, return_exceptions=True))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        with chroma_write_lock:
            final_count = vector_store.get_collection_count()

        logger.info(
            f"[{agent_name}] Done: collection={collection_name}, count={final_count}, "
            f"already_present={already_present}, newly_added={newly_added}"
        )
        return {
            "status": "ok",
            "collection_name": collection_name,
            "document_count": final_count,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "already_present": already_present,
            "newly_added": newly_added,
        }

    # Run agent builds concurrently.
    workers = max(1, min(int(max_workers), len(build_plan)))
    logger.info(f"Starting concurrent build: workers={workers}, agents={list(build_plan.keys())}")

    if workers == 1:
        for agent_name in build_plan.keys():
            try:
                results[agent_name] = _build_one_agent(agent_name)
            except Exception as e:
                logger.error(f"✗ Failed to build collection for {agent_name}: {str(e)}", exc_info=True)
                results[agent_name] = {
                    "status": "error",
                    "collection_name": str(build_plan[agent_name]["collection_name"]),
                    "error": str(e),
                    "embedding_model": all_agent_configs.get(agent_name, {}).get("embedding_model"),
                    "embedding_provider": all_agent_configs.get(agent_name, {}).get("embedding_provider"),
                }
                if not continue_on_error:
                    break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_build_one_agent, a): a for a in build_plan.keys()}
            for fut in as_completed(future_map):
                agent_name = future_map[fut]
                try:
                    results[agent_name] = fut.result()
                except Exception as e:
                    logger.error(f"✗ Failed to build collection for {agent_name}: {str(e)}", exc_info=True)
                    results[agent_name] = {
                        "status": "error",
                        "collection_name": str(build_plan[agent_name]["collection_name"]),
                        "error": str(e),
                        "embedding_model": all_agent_configs.get(agent_name, {}).get("embedding_model"),
                        "embedding_provider": all_agent_configs.get(agent_name, {}).get("embedding_provider"),
                    }
                    if not continue_on_error:
                        # Best-effort: cancel pending tasks.
                        for other in future_map:
                            if other is not fut:
                                other.cancel()
                        break

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
    parser.add_argument("--embedding-batch-size", dest="embedding_batch_size", type=int, default=10, help="embed batch size")
    parser.add_argument(
        "--embedding-concurrency",
        dest="embedding_concurrency",
        type=int,
        default=1,
        help="async OpenRouter embedding concurrency (default: 1; set >1 to enable per-agent async batching)",
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
        help="concurrent workers (default: 4; set 1 for sequential)",
    )
    parser.add_argument(
        "--sleep-between-batches",
        dest="sleep_between_batches",
        type=float,
        default=0.5,
        help="sleep seconds between embedding batches per agent (default: 0.5; set 0 to disable)",
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
    # When user explicitly passes --clear, prefer non-interactive clearing.
    clear_existing: Optional[bool] = True if args.clear_existing else None

    build_vector_databases_batch(
        config_path=args.config_path,
        data_dir=args.data_dir,
        literature_type_configs=LITERATURE_TYPE_CONFIGS,
        agent_names=agent_names,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        embedding_concurrency=args.embedding_concurrency,
        max_workers=args.max_workers,
        sleep_between_batches=args.sleep_between_batches,
        resume=bool(args.resume),
        clear_existing=clear_existing,
        skip_if_exists=bool(args.skip_if_exists),
        continue_on_error=not bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
