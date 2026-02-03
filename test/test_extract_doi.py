from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Allow running as a script: `python test/test_extract_doi.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from database.text_processor import TextProcessor


# A "real DOI" should always start with "10.<registrant>/".
_DOI_PREFIX_RE = re.compile(r"(?i)^10\.\d{4,9}/")


@dataclass
class ReactionStats:
    reaction: str
    total_files: int = 0
    with_doi: int = 0
    without_doi: int = 0
    unique_dois: int = 0


def _is_real_doi(doc_id: str) -> bool:
    return bool(_DOI_PREFIX_RE.match((doc_id or "").strip()))


def _extract_first_title_line(content: str) -> str:
    for line in (content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def main() -> int:
    """
    Scan `data/raw/<reaction>/*.md` and report how many files contain extractable DOIs.

    Notes:
    - TextProcessor.extract_doi_from_content returns either:
      - a DOI like "10.xxxx/....", or
      - a stable fallback "no-doi:..." when DOI is missing.
    - Therefore we count a "real DOI" only when doc_id matches '^10.\\d{4,9}/'.
    """

    repo_root = Path(__file__).resolve().parents[1]
    raw_root = repo_root / "data" / "raw"
    if not raw_root.exists():
        print(f"[ERROR] Directory not found: {raw_root}")
        return 1

    # Important: pass the dataset root so fallback IDs are stable and relative.
    processor = TextProcessor(data_dir=str(raw_root))

    reaction_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not reaction_dirs:
        print(f"[ERROR] No reaction directories found under: {raw_root}")
        return 1

    per_reaction: List[ReactionStats] = []
    global_total = 0
    global_with_doi = 0
    global_without_doi = 0

    # Keep a short debug list for missing DOI cases.
    missing_examples: List[Tuple[str, str, str]] = []

    for reaction_dir in reaction_dirs:
        reaction = reaction_dir.name
        md_files = sorted(reaction_dir.glob("*.md"))
        if not md_files:
            continue

        dois_seen: set[str] = set()
        stats = ReactionStats(reaction=reaction)

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            doc_id = processor.extract_doi_from_content(content, md_file.name, str(md_file))

            stats.total_files += 1
            global_total += 1

            if _is_real_doi(doc_id):
                stats.with_doi += 1
                global_with_doi += 1
                dois_seen.add(doc_id.lower().strip())
            else:
                stats.without_doi += 1
                global_without_doi += 1
                if len(missing_examples) < 25:
                    missing_examples.append((reaction, md_file.name, _extract_first_title_line(content)[:120]))

        stats.unique_dois = len(dois_seen)
        per_reaction.append(stats)

    # Report
    print("=" * 72)
    print(f"DOI Extraction Report (root: {raw_root})")
    print("=" * 72)

    if not per_reaction:
        print("[ERROR] No Markdown files found under any reaction directory.")
        return 1

    for s in per_reaction:
        print(
            f"{s.reaction:8s}  total={s.total_files:5d}  with_doi={s.with_doi:5d}  "
            f"without_doi={s.without_doi:5d}  unique_doi={s.unique_dois:5d}"
        )

    print("-" * 72)
    print(
        f"ALL      total={global_total:5d}  with_doi={global_with_doi:5d}  without_doi={global_without_doi:5d}"
    )

    if missing_examples:
        print("\nMissing DOI examples (first 25):")
        for reaction, fname, title in missing_examples:
            title_disp = title if title else "(no title line found)"
            # Avoid Windows console encoding issues (GBK) on some machines.
            title_safe = title_disp.encode("ascii", errors="replace").decode("ascii")
            print(f"- [{reaction}] {fname}  Title: {title_safe}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
