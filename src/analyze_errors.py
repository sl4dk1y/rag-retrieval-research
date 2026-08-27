"""Qualitative error analysis for BM25, Dense and Hybrid test runs.

The script compares the three frozen retrieval runs on the same Mr. TyDi split,
classifies queries by method success/failure, and writes query-level CSV plus a
summary JSON. It does not tune retrieval parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian" / "ir-format-data"

DEFAULT_BM25_RUN = PROJECT_ROOT / "results" / "runs" / "bm25_test.trec"
DEFAULT_DENSE_RUN = (
    PROJECT_ROOT / "results" / "runs" / "dense_e5_small_test_nprobe256.trec"
)
DEFAULT_HYBRID_RUN = (
    PROJECT_ROOT / "results" / "runs" / "hybrid_rrf_w0025_10_test.trec"
)

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "analysis" / "error_analysis_test.csv"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "results" / "analysis" / "error_analysis_test_summary.json"
)

DEFAULT_CUTOFF = 10


@dataclass(frozen=True)
class RunEntry:
    doc_id: str
    rank: int
    score: float


def find_topics_file(split: str) -> Path:
    candidates = sorted(DATASET_DIR.glob(f"topics.{split}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"Не найден topics для split={split!r} в {DATASET_DIR}"
        )
    return candidates[0]


def find_qrels_file(split: str) -> Path:
    candidates = sorted(DATASET_DIR.glob(f"qrels.{split}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"Не найден qrels для split={split!r} в {DATASET_DIR}"
        )
    return candidates[0]


def load_topics(split: str) -> dict[str, str]:
    path = find_topics_file(split)
    topics: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as stream:
        for number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Некорректная строка topics #{number} в {path}: {line!r}"
                )

            query_id, text = parts
            if query_id in topics:
                raise ValueError(f"Дубликат query_id={query_id!r} в {path}")
            topics[query_id] = text

    if not topics:
        raise ValueError(f"Файл topics пуст: {path}")

    return topics


def load_qrels(split: str) -> dict[str, dict[str, int]]:
    path = find_qrels_file(split)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    with path.open("r", encoding="utf-8") as stream:
        for number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 4:
                query_id, _iteration, doc_id, relevance_raw = parts
            elif len(parts) == 3:
                query_id, doc_id, relevance_raw = parts
            else:
                raise ValueError(
                    f"Некорректная строка qrels #{number} в {path}: {line!r}"
                )

            try:
                relevance = int(relevance_raw)
            except ValueError as error:
                raise ValueError(
                    f"Некорректная relevance в {path}, строка {number}"
                ) from error

            qrels[query_id][doc_id] = relevance

    if not qrels:
        raise ValueError(f"Файл qrels пуст: {path}")

    return dict(qrels)


def load_run(path: Path) -> dict[str, list[RunEntry]]:
    if not path.is_file():
        raise FileNotFoundError(f"TREC run не найден: {path}")

    runs: dict[str, list[RunEntry]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as stream:
        for number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"Некорректный TREC run {path}, строка {number}: "
                    f"ожидалось 6 колонок"
                )

            query_id, q0, doc_id, rank_raw, score_raw, _run_name = parts
            if q0 != "Q0":
                raise ValueError(
                    f"Вторая колонка должна быть Q0 в {path}, строка {number}"
                )

            try:
                rank = int(rank_raw)
                score = float(score_raw)
            except ValueError as error:
                raise ValueError(
                    f"Некорректный rank/score в {path}, строка {number}"
                ) from error

            if rank <= 0:
                raise ValueError(f"Rank должен быть > 0 в {path}, строка {number}")
            if doc_id in seen[query_id]:
                raise ValueError(
                    f"Дубликат doc_id={doc_id!r} для query={query_id!r} в {path}"
                )

            seen[query_id].add(doc_id)
            runs[query_id].append(RunEntry(doc_id=doc_id, rank=rank, score=score))

    for entries in runs.values():
        entries.sort(key=lambda item: item.rank)

    return dict(runs)


def best_relevant_rank(
    entries: Iterable[RunEntry],
    relevant_docids: set[str],
    cutoff: int,
) -> int | None:
    ranks = [
        entry.rank
        for entry in entries
        if entry.rank <= cutoff and entry.doc_id in relevant_docids
    ]
    return min(ranks) if ranks else None


def top_docids(entries: Iterable[RunEntry], limit: int = 5) -> str:
    return " | ".join(entry.doc_id for entry in list(entries)[:limit])


def classify(
    bm25_rank: int | None,
    dense_rank: int | None,
    hybrid_rank: int | None,
) -> str:
    bm25_ok = bm25_rank is not None
    dense_ok = dense_rank is not None
    hybrid_ok = hybrid_rank is not None

    if bm25_ok and dense_ok and hybrid_ok:
        return "all_success"
    if not bm25_ok and not dense_ok and not hybrid_ok:
        return "all_fail"

    if bm25_ok and not dense_ok:
        return "bm25_success_dense_fail"
    if dense_ok and not bm25_ok:
        if hybrid_ok:
            return "dense_success_bm25_fail"
        return "dense_success_hybrid_fail"

    if hybrid_ok and not dense_ok:
        return "hybrid_recovers_dense_failure"
    if dense_ok and not hybrid_ok:
        return "hybrid_hurts_dense"

    return "mixed_other"


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--bm25-run", type=Path, default=DEFAULT_BM25_RUN)
    parser.add_argument("--dense-run", type=Path, default=DEFAULT_DENSE_RUN)
    parser.add_argument("--hybrid-run", type=Path, default=DEFAULT_HYBRID_RUN)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON
    )
    args = parser.parse_args()

    if args.cutoff <= 0:
        raise ValueError("--cutoff должен быть > 0")

    topics = load_topics(args.split)
    qrels = load_qrels(args.split)
    bm25 = load_run(args.bm25_run)
    dense = load_run(args.dense_run)
    hybrid = load_run(args.hybrid_run)

    query_ids = set(topics)
    for name, run in (("BM25", bm25), ("Dense", dense), ("Hybrid", hybrid)):
        if set(run) != query_ids:
            missing = sorted(query_ids - set(run))
            extra = sorted(set(run) - query_ids)
            raise ValueError(
                f"{name} run имеет другой набор запросов. "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    rows: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    pairwise_counts: Counter[str] = Counter()

    for query_id in sorted(query_ids):
        relevant_docids = {
            doc_id
            for doc_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }

        bm25_rank = best_relevant_rank(bm25[query_id], relevant_docids, args.cutoff)
        dense_rank = best_relevant_rank(dense[query_id], relevant_docids, args.cutoff)
        hybrid_rank = best_relevant_rank(hybrid[query_id], relevant_docids, args.cutoff)

        category = classify(bm25_rank, dense_rank, hybrid_rank)
        category_counts[category] += 1

        dense_rr = reciprocal_rank(dense_rank)
        hybrid_rr = reciprocal_rank(hybrid_rank)

        if hybrid_rr > dense_rr:
            pairwise_counts["hybrid_better_than_dense"] += 1
        elif hybrid_rr < dense_rr:
            pairwise_counts["dense_better_than_hybrid"] += 1
        else:
            pairwise_counts["dense_hybrid_equal"] += 1

        rows.append(
            {
                "query_id": query_id,
                "query": topics[query_id],
                "relevant_docs": len(relevant_docids),
                f"bm25_best_rel_rank@{args.cutoff}": bm25_rank or "",
                f"dense_best_rel_rank@{args.cutoff}": dense_rank or "",
                f"hybrid_best_rel_rank@{args.cutoff}": hybrid_rank or "",
                "category": category,
                "dense_vs_hybrid": (
                    "hybrid_better"
                    if hybrid_rr > dense_rr
                    else "dense_better"
                    if hybrid_rr < dense_rr
                    else "equal"
                ),
                "bm25_top5_docids": top_docids(bm25[query_id]),
                "dense_top5_docids": top_docids(dense[query_id]),
                "hybrid_top5_docids": top_docids(hybrid[query_id]),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "query_count": len(rows),
        "cutoff": args.cutoff,
        "category_counts": dict(sorted(category_counts.items())),
        "dense_vs_hybrid": dict(sorted(pairwise_counts.items())),
        "runs": {
            "bm25": str(args.bm25_run.relative_to(PROJECT_ROOT)),
            "dense": str(args.dense_run.relative_to(PROJECT_ROOT)),
            "hybrid": str(args.hybrid_run.relative_to(PROJECT_ROOT)),
        },
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Error analysis ===")
    print(f"Split:         {args.split}")
    print(f"Queries:       {len(rows):,}")
    print(f"Cutoff:        @{args.cutoff}")
    print()
    print("Categories:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:32s} {count:4d}")
    print()
    print("Dense vs Hybrid:")
    for category, count in sorted(pairwise_counts.items()):
        print(f"  {category:32s} {count:4d}")
    print()
    print(f"CSV saved:     {args.output_csv}")
    print(f"Summary saved: {args.summary_json}")


if __name__ == "__main__":
    main()
