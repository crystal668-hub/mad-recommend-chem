"""
===================================
Preprocessing Module 
Function: Process raw text data and prepare for vectorization
Includes text loading, chunking, and supports LlamaIndex parsing of Markdown files
===================================
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from utils.logger import Logger

if TYPE_CHECKING:
    # Optional: only required when calling `load_documents` / `chunk_documents`.
    from llama_index.core import Document

logger = Logger.create_module_logger("database.text_processor")


# DOI extraction patterns (Markdown often wraps DOI in links/punctuation).
# We keep the suffix permissive and normalize/validate after matching.
_DOI_URL_RE = re.compile(
    r"(?i)\b(?:https?://)?(?:dx\.)?doi\.org/(?P<doi>10\.\d{4,9}/[^\s<>\"]+)"
)
_DOI_LABEL_RE = re.compile(
    r"(?i)\bdoi\s*[:：]?\s*(?P<doi>10\.\d{4,9}/[^\s<>\"]+)"
)
_DOI_BARE_RE = re.compile(r"(?i)\b(?P<doi>10\.\d{4,9}/[^\s<>\"]+)")

_REACTION_TYPES = {"co2rr", "eor", "her", "hor", "hzor", "o5h", "oer", "orr", "uor"}


class TextProcessor:
    """
    Responsible for loading and processing raw document data
    """
    
    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize the text processor
        
        Args:
            data_dir: Directory containing raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded mapping: reaction_key -> {normalized_title: doi}
        self._metadata_tsv_index: Dict[str, Dict[str, str]] = {}
        self._metadata_tsv_paths: Optional[Dict[str, List[Path]]] = None
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本内容，删除Acknowledgement和Reference部分
        
        清洗操作：
        - 删除Acknowledgement部分及其后的所有内容
        - 删除Reference/References部分及其后的所有内容
        - 若无Reference标题，尝试移除末尾参考文献列表
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""
        
        # 删除Acknowledgement部分及其后的内容
        # 匹配各种可能的写法：Acknowledgement, Acknowledgements, ACKNOWLEDGEMENT等
        text = re.sub(
            r'(?i)(##?\s*)?Acknowledgements?.*$',
            '',
            text,
            flags=re.DOTALL
        )
        
        # 删除Reference/References部分及其后的内容
        # 匹配各种可能的写法：Reference, References, REFERENCE, REFERENCES等
        text = re.sub(
            r'(?i)(##?\s*)?References?.*$',
            '',
            text,
            flags=re.DOTALL
        )

        # 若无Reference标题，尝试识别并删除末尾参考文献列表
        # 典型格式：
        # [1] Author, ... 2021 ...
        # 1. Author, ... 2021 ...
        # 1) Author, ... 2021 ...
        def _is_reference_line(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            if re.search(r'^\[\d+\]\s+\S', stripped):
                return True
            if re.search(r'^\d{1,3}[\.)]\s+\S', stripped):
                return True
            if re.search(r'\bdoi\b', stripped, re.IGNORECASE):
                return True
            if re.search(r'\b(19|20)\d{2}[a-z]?\b', stripped):
                if stripped.count(',') >= 2 or stripped.count('.') >= 2:
                    return True
            return False

        lines = text.splitlines()
        end_index = len(lines) - 1
        while end_index >= 0 and not lines[end_index].strip():
            end_index -= 1

        if end_index > 0:
            ref_lines = 0
            block_start = None
            for i in range(end_index, -1, -1):
                line = lines[i]
                if not line.strip():
                    if ref_lines > 0:
                        block_start = i
                    continue
                if _is_reference_line(line):
                    ref_lines += 1
                    block_start = i
                else:
                    if ref_lines >= 3:
                        break
                    ref_lines = 0
                    block_start = None

            if ref_lines >= 3 and block_start is not None:
                text = "\n".join(lines[:block_start]).rstrip()
        
        return text

    def _prepare_loaded_document_text(self, doc: Document) -> str:
        """
        Clean a loaded Markdown document in-place before metadata extraction/chunking.

        This ensures downstream chunking sees the cleaned paper body rather than
        trailing acknowledgements or reference sections.
        """
        raw_text = getattr(doc, "text", "") or ""
        cleaned_text = self.clean_text(raw_text)
        if cleaned_text != raw_text:
            doc.set_content(cleaned_text)
        return cleaned_text

    @staticmethod
    def _normalize_title_for_match(text: str) -> str:
        """
        Normalize a paper title for robust matching across Markdown/TSV sources.

        Strategy: lowercase, remove HTML tags/markdown emphasis, collapse punctuation to spaces.
        """
        if not text:
            return ""
        t = text.strip().lower()
        # Remove HTML tags like <sub>2</sub>, <sup>+</sup>.
        t = re.sub(r"<[^>]+>", "", t)
        # Drop common markdown emphasis markers.
        t = t.replace("**", " ").replace("*", " ").replace("_", " ").replace("`", " ")
        # Collapse non-word characters to spaces (keep ASCII alnum + CJK).
        t = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _extract_title_candidates_from_markdown(content: str) -> List[str]:
        """
        Extract plausible title candidates from a Markdown file.

        Heuristics (in priority order):
        - Prefer the first *non-generic* H1 (# ...) title near the top.
        - Skip common section/journal headers like "FULL PAPER", "RESEARCH ARTICLE", "ABSTRACT", etc.
        - If needed, fall back to a "Title:" marker or bold-only lines.
        """
        if not content:
            return []

        lines = content.splitlines()
        head_lines = lines[:120]

        noise_exact = {
            "full paper",
            "paper",
            "short note",
            "selected paper",
            "research article",
            "research articles",
            "original article",
            "original research",
            "review article",
            "accepted article",
            "open",
            "open access",
            "abstract",
            "keywords",
            "keyword",
            "correspondence",
            "funding",
            "funding information",
            "author contributions",
            "acknowledgment",
            "acknowledgments",
            "acknowledgement",
            "acknowledgements",
            "introduction",
            "conclusion",
            "conclusions",
            "references",
            "reference",
            "zuschriften",
            "forschungsartikel",
        }
        noise_patterns = [
            re.compile(r"(?i)^(original|research|review)\s+article\b"),
            re.compile(r"(?i)^.*open\s+access.*$"),
            re.compile(r"(?i)^\d+(\.\d+)?\s*\|\s*\w+.*$"),  # e.g. "1 | INTRODUCTION"
            re.compile(r"(?i)^(figure|scheme|table)\b"),
        ]

        def _strip_markdown(text: str) -> str:
            t = (text or "").strip()
            if not t:
                return ""
            # Remove outer emphasis markers and inline formatting.
            t = re.sub(r"^\*+|\*+$", "", t).strip()
            t = re.sub(r"`+", "", t)
            t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
            t = re.sub(r"_([^_]+)_", r"\1", t)
            return t.strip()

        def _is_noise_title(text: str) -> bool:
            plain = _strip_markdown(text)
            if not plain:
                return True
            norm = TextProcessor._normalize_title_for_match(plain)
            if not norm:
                return True
            if norm in noise_exact:
                return True
            for rx in noise_patterns:
                if rx.match(plain) or rx.match(norm):
                    return True

            # Many exports put section/journal headers in ALL CAPS; treat short ALL-CAPS as noise.
            letters = re.sub(r"[^A-Za-z]+", "", plain)
            if letters and letters.isupper():
                words = plain.split()
                if len(words) <= 4 and len(plain) <= 40:
                    return True

            return False

        candidates: List[str] = []

        # Pattern 1: "Title:" marker (sometimes present in structured exports).
        for line in head_lines:
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"(?i)^title\s*[:：]\s*(.+)$", stripped)
            if not m:
                continue
            title = _strip_markdown(m.group(1))
            if title and not _is_noise_title(title):
                candidates.append(title)

        # Pattern 2: Markdown headings; prefer H1 and skip noisy headers like "FULL PAPER".
        headings: List[tuple[int, str]] = []
        for line in head_lines:
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if not m:
                continue
            level = len(m.group(1))
            title = _strip_markdown(m.group(2))
            if not title or _is_noise_title(title):
                continue
            headings.append((level, title))

        # Preserve document order, but prioritize lower heading level (H1 -> H6).
        for wanted_level in range(1, 7):
            for level, title in headings:
                if level != wanted_level:
                    continue
                if title not in candidates:
                    candidates.append(title)
                if len(candidates) >= 5:
                    return candidates

        # Pattern 3: bold-only lines (last resort).
        for line in head_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                title = _strip_markdown(stripped)
                if title and len(title) >= 15 and not _is_noise_title(title):
                    candidates.append(title)
                    break

        return candidates

    @staticmethod
    def _build_title_variants(title: str) -> List[str]:
        """Build a small set of variants to improve exact-match recall."""
        t = (title or "").strip()
        if not t:
            return []
        variants = [t]
        for sep in (":", " - ", " – ", " — "):
            if sep in t:
                variants.append(t.split(sep, 1)[0].strip())
        # De-dup while preserving order.
        seen = set()
        out: List[str] = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    @staticmethod
    def _get_reaction_key_from_path(file_path: str) -> Optional[str]:
        if not file_path:
            return None
        parts = [p.lower() for p in Path(file_path).parts]
        for part in parts:
            if part in _REACTION_TYPES:
                return part
        return None

    def _get_metadata_tsv_paths(self) -> Dict[str, List[Path]]:
        """
        Map reaction_key -> list of TSV paths under ./metadata.

        TSV files in this repo are large; we only load the reaction we need on demand.
        """
        if self._metadata_tsv_paths is not None:
            return self._metadata_tsv_paths

        metadata_dir = Path("./metadata")
        paths: Dict[str, List[Path]] = {}
        if metadata_dir.exists():
            for tsv_path in metadata_dir.glob("*.tsv"):
                name = tsv_path.name.lower()
                reaction_key = None
                for rt in _REACTION_TYPES:
                    if rt in name:
                        reaction_key = rt
                        break
                if reaction_key is None:
                    continue
                paths.setdefault(reaction_key, []).append(tsv_path)

        self._metadata_tsv_paths = paths
        return paths

    def _load_metadata_tsv_index(self, reaction_key: str) -> Dict[str, str]:
        """
        Load and cache {normalized_title: doi} for a given reaction TSV metadata file.

        Expected TSV schema includes at least: title, doi
        """
        rk = (reaction_key or "").strip().lower()
        if not rk:
            return {}
        if rk in self._metadata_tsv_index:
            return self._metadata_tsv_index[rk]

        paths_by_reaction = self._get_metadata_tsv_paths()
        tsv_paths = paths_by_reaction.get(rk) or []
        if not tsv_paths:
            self._metadata_tsv_index[rk] = {}
            return self._metadata_tsv_index[rk]

        index: Dict[str, str] = {}
        for tsv_path in tsv_paths:
            try:
                logger.info(f"Loading metadata TSV for {rk}: {tsv_path.name}")
                with tsv_path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
                    header = handle.readline()
                    if not header:
                        continue
                    columns = [c.strip().lower() for c in header.rstrip("\n").split("\t")]

                    def _find_col(*names: str) -> Optional[int]:
                        for n in names:
                            if n in columns:
                                return columns.index(n)
                        return None

                    title_idx = _find_col("title", "paper_title", "article_title")
                    doi_idx = _find_col("doi", "doi_url", "doi link", "doi_link")
                    if title_idx is None or doi_idx is None:
                        logger.warning(f"TSV missing required columns (title/doi): {tsv_path.name}")
                        continue

                    for line in handle:
                        if not line.strip():
                            continue
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) <= max(title_idx, doi_idx):
                            continue
                        title = (parts[title_idx] or "").strip()
                        doi_raw = (parts[doi_idx] or "").strip()
                        if not title or not doi_raw:
                            continue
                        doi = self._normalize_doi(doi_raw) or self._normalize_doi(doi_raw.replace("https://doi.org/", ""))
                        if not doi:
                            continue
                        key = self._normalize_title_for_match(title)
                        if key and key not in index:
                            index[key] = doi
            except Exception as exc:
                logger.error(f"Failed to load TSV metadata: {tsv_path} - {exc}")

        self._metadata_tsv_index[rk] = index
        return index

    def _extract_doi_from_tsv_metadata(self, content: str, file_path: str = "") -> Optional[str]:
        """Fallback: match Markdown title to metadata TSV (title -> doi)."""
        reaction_key = self._get_reaction_key_from_path(file_path) if file_path else None
        if not reaction_key:
            return None

        titles = self._extract_title_candidates_from_markdown(content)
        if not titles:
            return None

        title_map = self._load_metadata_tsv_index(reaction_key)
        if not title_map:
            return None

        for title in titles:
            for variant in self._build_title_variants(title):
                key = self._normalize_title_for_match(variant)
                if not key:
                    continue
                doi = title_map.get(key)
                if doi:
                    return doi

        return None

    def _load_metadata_csv_index(self, metadata_csv_path: str) -> Dict[str, Dict[str, str]]:
        """Load CSV metadata keyed by the case-insensitive PDF filename stem."""
        csv_path = Path(metadata_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv_path}")
        if not csv_path.is_file():
            raise ValueError(f"Metadata CSV path is not a file: {metadata_csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header_lookup = {
                str(name or "").strip(): name
                for name in (reader.fieldnames or [])
            }
            required_headers = {"file_name", "doi", "abstract"}
            missing = sorted(required_headers - set(header_lookup))
            if missing:
                raise ValueError(
                    f"Metadata CSV missing required columns: {', '.join(missing)}"
                )

            index: Dict[str, Dict[str, str]] = {}
            for row in reader:
                file_name = str(row.get(header_lookup["file_name"]) or "").strip()
                if not file_name:
                    continue
                key = Path(file_name).stem.casefold()
                if key in index:
                    raise ValueError(
                        f"Duplicate file_name basename in metadata CSV: {key}"
                    )
                index[key] = {
                    "file_name": file_name,
                    "doi": str(row.get(header_lookup["doi"]) or "").strip(),
                    "abstract": str(row.get(header_lookup["abstract"]) or "").strip(),
                }

        return index
    
    @staticmethod
    def _normalize_doi(raw: str) -> Optional[str]:
        """Normalize an extracted DOI candidate into canonical `10.xxxx/...` form."""
        if not raw:
            return None

        doi = raw.strip()

        # Strip common URL / label prefixes (defensive; regexes usually capture DOI only).
        doi = re.sub(r"(?i)^(?:doi\s*[:：]\s*)", "", doi)
        doi = re.sub(r"(?i)^(?:https?://)?(?:dx\.)?doi\.org/", "", doi)

        # Remove query/fragment if DOI came from a URL.
        doi = doi.split("?", 1)[0].split("#", 1)[0]

        # Strip surrounding wrappers often introduced by Markdown/prose.
        doi = doi.strip().strip("<>")
        doi = doi.lstrip("([{<\"'")
        doi = doi.rstrip(".,;:!?\"'")
        # Markdown emphasis wrappers can be captured by permissive DOI regexes, e.g. **10.xxxx/...**
        # Remove them so doc_id/source_id stays canonical.
        doi = doi.strip("*_`")

        # Guardrails for Markdown artifacts:
        # Some marker-generated Markdown contains patterns like:
        #   [https://doi.org/10.3390/](https://doi.org/10.3390/catal15080767)
        # Our permissive DOI regex can accidentally capture "10.3390/](https://...)" (no whitespace),
        # which is NOT a valid DOI and also bloats metadata, crashing metadata-aware chunking.
        low = doi.lower()
        if "](" in low or "](http" in low or "doi.org" in low or "http://" in low or "https://" in low:
            return None

        # Remove unbalanced closing brackets at the end, e.g. from Markdown links "...(doi)".
        closers = {")": "(", "]": "[", "}": "{", ">": "<"}
        while doi and doi[-1] in closers and doi.count(closers[doi[-1]]) < doi.count(doi[-1]):
            doi = doi[:-1]
            doi = doi.rstrip(".,;:!?\"'")

        doi = doi.strip()
        if not doi:
            return None

        # Basic validation (keep permissive; DOI suffix allows many characters but no whitespace).
        if re.search(r"\s", doi):
            return None
        # Require at least one suffix character after the slash.
        if not re.match(r"(?i)^10\.\d{4,9}/\S+", doi):
            return None

        # Canonicalize for stable IDs in metadata / source_id.
        return doi.lower()

    def _extract_doi_from_markdown(self, content: str) -> Optional[str]:
        """Extract DOI from Markdown content (plain form or doi.org URL)."""
        if not content:
            return None

        # Avoid picking up DOIs from reference lists at the end of the document.
        cleaned = self.clean_text(content)

        # Most papers place DOI in the header area; scan head first (more reliable).
        head = cleaned[:8000]
        for regex in (_DOI_URL_RE, _DOI_LABEL_RE, _DOI_BARE_RE):
            for match in regex.finditer(head):
                doi = self._normalize_doi(match.group("doi"))
                if doi:
                    return doi

        # Fallback: scan a larger window for explicit DOI/url forms (still avoid bare matches).
        window = cleaned[:20000]
        for regex in (_DOI_URL_RE, _DOI_LABEL_RE):
            for match in regex.finditer(window):
                doi = self._normalize_doi(match.group("doi"))
                if doi:
                    return doi

        return None

    def _build_fallback_doc_id(self, filename: str = "", file_path: str = "") -> str:
        """
        构造一个稳定的“非DOI”文档标识，用于 doc_id/source_id。

        说明：
        - 下游会用 (doc_id, chunk_id) 拼接 source_id；若大量返回 "unknown"，会产生跨文档冲突。
        - 因此在没有 DOI 的情况下，优先用“相对 data_dir 的路径”生成一个稳定 ID。
        """
        candidate = ""

        # Prefer a path relative to the dataset root for stability across machines.
        if file_path:
            try:
                p = Path(file_path)
                try:
                    rel = p.resolve().relative_to(self.data_dir.resolve())
                    candidate = rel.as_posix()
                except Exception:
                    candidate = p.name
            except Exception:
                candidate = str(file_path)
        elif filename:
            candidate = str(filename)

        candidate = (candidate or "").strip().replace("\\", "/")
        if not candidate:
            return "unknown"

        # Drop extension and sanitize to keep the ID URL-safe-ish.
        candidate = re.sub(r"(?i)\.md$", "", candidate)
        candidate = re.sub(r"\s+", "_", candidate)
        candidate = re.sub(r"[^0-9A-Za-z._/-]+", "_", candidate).strip("_")

        # IMPORTANT:
        # LlamaIndex's metadata-aware chunking counts metadata tokens against `chunk_size`.
        # If users name markdown files with very long titles, keeping the full filename/path
        # here can make `doc_id` exceed the chunk token budget and crash chunking.
        #
        # To keep metadata bounded while still being stable + partially readable, we use:
        #   no-doi:<basename_trunc>_<sha16>
        slug = candidate.lower()
        basename = slug.rsplit("/", 1)[-1] or "doc"
        basename = basename[:30].strip("_") or "doc"
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:16]
        return f"no-doi:{basename}_{digest}"

    def extract_doi_from_content(self, content: str, filename: str = "", file_path: str = "") -> str:
        """
        提取并返回DOI。

        优先从Markdown正文中直接解析DOI（支持显式 DOI: 10.xxxx/... 以及 doi.org URL 形式）。
        若正文未找到，则回退到 ./metadata/*.tsv 中按 title->doi 匹配提取。
        若仍未找到，则返回稳定的 fallback 标识（no-doi:...），避免 doc_id="unknown" 导致 source_id 冲突。
         
        Args:
            content: 文本内容
            filename: 文件名
            file_path: 文件路径
            
        Returns:
            str: DOI；如果未找到则返回稳定的 fallback 标识（no-doi:...）
        """
        doi = self._extract_doi_from_markdown(content)
        if doi:
            return doi

        doi = self._extract_doi_from_tsv_metadata(content, file_path=file_path)
        if doi:
            return doi

        return self._build_fallback_doc_id(filename=filename, file_path=file_path)
    
    def load_documents(
        self,
        data_dir: str,
        reaction_type: Optional[str] = None
    ) -> List[Document]:
        """
        使用SimpleDirectoryReader加载Markdown文件，创建Document对象并添加metadata
        
        Args:
            data_dir: 数据目录路径
            reaction_type: 反应类型
        
        Returns:
            List[Document]: Document对象列表
            metadata包含：
                - reaction_type: 所属反应类型
                - doc_id: 文献DOI号
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Directory does not exist: {data_dir}")
            return []
        
        try:
            # Local import: DOI-only scripts shouldn't require LlamaIndex installed.
            from llama_index.core import SimpleDirectoryReader

            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                required_exts=[".md"],  
                recursive=False  
            )
            documents = reader.load_data()
            
            if not documents:
                logger.error(f"No Markdown files found: {data_dir}")
                return []
            
            # Add custom metadata to each document
            processed_documents = []
            for doc in documents:
                cleaned_text = self._prepare_loaded_document_text(doc)

                # Get file name (from original metadata)
                file_name = doc.metadata.get('file_name', '')
                file_path_str = doc.metadata.get('file_path', '')
                
                # Infer reaction type from file name (if not specified)
                inferred_reaction = reaction_type
                if inferred_reaction is None and file_name:
                    # Try to extract reaction type from file name
                    filename_upper = Path(file_name).stem.upper()
                    reaction_types = ["CO2RR", "EOR", "HER", "HOR", "HZOR", "O5H", "OER", "ORR", "UOR"]
                    for rt in reaction_types:
                        if rt in filename_upper:
                            inferred_reaction = rt
                            break
                
                # Extract DOI 
                doc_id = self.extract_doi_from_content(cleaned_text, file_name, file_path_str)
                
                # Reset metadata to only include reaction_type and doc_id
                doc.metadata = {
                    "reaction_type": inferred_reaction or "unknown",
                    "doc_id": doc_id
                }
                
                processed_documents.append(doc)
                
                logger.info(f" Loaded {file_name}")
                logger.info(f" Reaction Type: {doc.metadata['reaction_type']}")
                logger.info(f" DOI: {doc.metadata['doc_id']}")
            
            logger.info(f"\nLoaded a total of {len(processed_documents)} Document objects")
            return processed_documents
            
        except Exception as e:
            logger.error(f" Failed to load documents: {str(e)}")
            return []

    def load_literature_type_documents(
        self,
        base_dir: str,
        literature_type_configs: Dict[str, Dict],
    ) -> List[Document]:
        """Load every configured Literature Type through the CSV-backed workflow."""
        all_documents: List[Document] = []
        base_path = Path(base_dir)

        for literature_type, config in literature_type_configs.items():
            literature_path = base_path / config.get("path", literature_type)
            logger.info(f"\n--- Loading {literature_type} ---")
            docs = self.load_literature_type_directory_documents(
                data_dir=str(literature_path),
                metadata_csv_path=str(config["metadata_csv"]),
                literature_type=literature_type,
            )
            all_documents.extend(docs)

        return all_documents

    def load_literature_type_directory_documents(
        self,
        data_dir: str,
        metadata_csv_path: str,
        literature_type: str,
    ) -> List[Document]:
        """Load one Literature Type directory and resolve metadata from its CSV."""
        metadata_index = self._load_metadata_csv_index(metadata_csv_path)
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Directory does not exist: {data_dir}")
            return []

        md_files = sorted(path for path in data_path.glob("*.md") if path.is_file())
        if not md_files:
            logger.error(f"No top-level Markdown files found: {data_dir}")
            return []

        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(path) for path in md_files],
            required_exts=[".md"],
            recursive=False,
        )
        documents = reader.load_data()
        if not documents:
            logger.error(f"No Markdown files loaded: {data_dir}")
            return []

        processed_documents: List[Document] = []
        matched_keys: set[str] = set()
        resolved_type = (literature_type or "unknown").strip() or "unknown"

        for doc in documents:
            cleaned_text = self._prepare_loaded_document_text(doc)
            file_name = doc.metadata.get("file_name", "")
            file_path_str = doc.metadata.get("file_path", "")
            file_stem = Path(file_name).stem if file_name else Path(file_path_str).stem
            metadata_key = file_stem.casefold()
            metadata_row = metadata_index.get(metadata_key)
            doc_id = None

            if metadata_row is None:
                logger.warning(
                    f"No CSV metadata mapping for {resolved_type}/{file_stem}; "
                    "falling back to Markdown DOI or no-doi id."
                )
            else:
                matched_keys.add(metadata_key)
                doc_id = self._normalize_doi(metadata_row.get("doi", ""))
                if not doc_id:
                    logger.warning(
                        f"Missing or invalid CSV DOI for {resolved_type}/{file_stem}; "
                        "falling back to Markdown DOI or no-doi id."
                    )

            if not doc_id:
                doc_id = self._extract_doi_from_markdown(cleaned_text)
            if not doc_id:
                doc_id = self._build_fallback_doc_id(file_name, file_path_str)

            doc.metadata = {
                "reaction_type": resolved_type,
                "doc_id": doc_id,
            }
            processed_documents.append(doc)

            logger.info(f" Loaded {file_name}")
            logger.info(f" Literature Type: {resolved_type}")
            logger.info(f" DOI: {doc_id}")

        extra_keys = sorted(set(metadata_index) - matched_keys)
        if extra_keys:
            examples = ", ".join(extra_keys[:10])
            logger.warning(
                f"Metadata CSV has {len(extra_keys)} extra row(s) without corresponding Markdown files: {examples}"
            )

        logger.info(
            f"\nLoaded a total of {len(processed_documents)} Document objects for {resolved_type}"
        )
        return processed_documents
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 256,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        使用 “两级切分策略” 对Document对象进行分块
        
        两级切分策略：
        1. 第一级：使用MarkdownNodeParser基于Markdown标题切分，确保chunk不跨越大章节
        2. 第二级：使用SentenceSplitter对第一级节点进行细分
        3. 使用IngestionPipeline将两级切分器串联
        
        Args:
            documents: Document对象列表
            chunk_size: 分块大小（默认256）
            chunk_overlap: 分块重叠大小（默认50）
        
        Returns:
            List[Document]: 分块后的Document列表
        """
        if not documents:
            return []

        # Local import: keep the module importable even when LlamaIndex isn't installed,
        # as long as callers don't use chunking/loading.
        from llama_index.core import Document
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
        
        # 第一级：Splitter Based on Markdown Title 
        markdown_parser = MarkdownNodeParser(include_metadata=False)
        
        # 第二级：Splitter Based on Sentence 
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=False
        )
        
        # 使用IngestionPipeline串联两级切分器
        pipeline = IngestionPipeline(
            transformations=[
                markdown_parser,      
                sentence_splitter     
            ]
        )
        
        chunked_documents = []
        skipped_empty_chunks = 0
        
        for doc in documents:
            nodes = pipeline.run(documents=[doc])
            
            for idx, node in enumerate(nodes):
                content = node.get_content()
                if not content or not str(content).strip():
                    skipped_empty_chunks += 1
                    continue

                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_id"] = idx
                chunk_metadata["total_chunks"] = len(nodes)
                
                chunk_doc = Document(
                    text=content,
                    metadata=chunk_metadata
                )
                chunked_documents.append(chunk_doc)
        
        if skipped_empty_chunks:
            logger.info(
                f"\nChunk Completed: {len(documents)} documents -> {len(chunked_documents)} chunks "
                f"(skipped {skipped_empty_chunks} empty chunks)"
            )
        else:
            logger.info(f"\nChunk Completed: {len(documents)} documents -> {len(chunked_documents)} chunks")
        return chunked_documents
