"""Combine BM25 and Dense TREC runs with weighted Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BM25_RUN = PROJECT_ROOT / "results" / "runs" / "bm25_dev.trec"
DEFAULT_DENSE_RUN = (
    PROJECT_ROOT / "results" / "runs" / "dense_e5_small_dev_nprobe256.trec"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "runs" / "hybrid_rrf_dev.trec"

DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 100
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_DENSE_WEIGHT = 1.0


@dataclass(frozen=True)
class RunEntry:
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_name: str


def parse_trec_run(path: Path) -> dict[str, list[RunEntry]]:
    """Read a six-column TREC run and validate ranks/documents per query."""
    if not path.is_file():
        raise FileNotFoundError(f"TREC run не найден: {path}")

    runs: dict[str, list[RunEntry]] = defaultdict(list)
    seen_docs: dict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"Некорректный TREC run в {path}, строка {line_number}: "
                    f"ожидалось 6 колонок, получено {len(parts)}"
                )

            query_id, q0, doc_id, rank_raw, score_raw, run_name = parts

            if q0 != "Q0":
                raise ValueError(
                    f"Некорректная вторая колонка в {path}, строка {line_number}: "
                    f"{q0!r}; ожидалось 'Q0'"
                )

            try:
                rank = int(rank_raw)
            except ValueError as error:
                raise ValueError(
                    f"Некорректный rank в {path}, строка {line_number}: {rank_raw!r}"
                ) from error

            try:
                score = float(score_raw)
            except ValueError as error:
                raise ValueError(
                    f"Некорректный score в {path}, строка {line_number}: {score_raw!r}"
                ) from error

            if rank <= 0:
                raise ValueError(
                    f"Rank должен быть положительным в {path}, строка {line_number}"
                )

            if doc_id in seen_docs[query_id]:
                raise ValueError(
                    f"Дубликат doc_id={doc_id!r} для query_id={query_id!r} "
                    f"в {path}"
                )

            seen_docs[query_id].add(doc_id)
            runs[query_id].append(
                RunEntry(
                    query_id=query_id,
                    doc_id=doc_id,
                    rank=rank,
                    score=score,
                    run_name=run_name,
                )
            )

    if not runs:
        raise ValueError(f"TREC run пуст: {path}")

    for query_id, entries in runs.items():
        entries.sort(key=lambda item: item.rank)
        expected = list(range(1, len(entries) + 1))
        actual = [entry.rank for entry in entries]

        if actual != expected:
            raise ValueError(
                f"Непоследовательные ranks для query_id={query_id!r} в {path}. "
                f"Ожидалось 1..{len(entries)}"
            )

    return dict(runs)


def rrf_score(rank: int, k: int) -> float:
    """Classic Reciprocal Rank Fusion contribution for one ranked list."""
    return 1.0 / (k + rank)


def fuse_query(
    bm25_entries: Iterable[RunEntry],
    dense_entries: Iterable[RunEntry],
    *,
    rrf_k: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> list[tuple[str, float]]:
    """Fuse BM25 and Dense rankings for one query using weighted RRF."""
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}

    for entry in bm25_entries:
        scores[entry.doc_id] += bm25_weight * rrf_score(entry.rank, rrf_k)
        previous = best_rank.get(entry.doc_id)
        if previous is None or entry.rank < previous:
            best_rank[entry.doc_id] = entry.rank

    for entry in dense_entries:
        scores[entry.doc_id] += dense_weight * rrf_score(entry.rank, rrf_k)
        previous = best_rank.get(entry.doc_id)
        if previous is None or entry.rank < previous:
            best_rank[entry.doc_id] = entry.rank

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            best_rank[item[0]],
            item[0],
        ),
    )

    return ranked[:top_k]


def fuse_runs(
    bm25_run: dict[str, list[RunEntry]],
    dense_run: dict[str, list[RunEntry]],
    *,
    rrf_k: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse two complete runs and require identical query sets."""
    bm25_queries = set(bm25_run)
    dense_queries = set(dense_run)

    if bm25_queries != dense_queries:
        only_bm25 = sorted(bm25_queries - dense_queries)
        only_dense = sorted(dense_queries - bm25_queries)
        raise ValueError(
            "BM25 и Dense run содержат разные query_id. "
            f"Только BM25: {only_bm25[:10]}; "
            f"только Dense: {only_dense[:10]}"
        )

    fused: dict[str, list[tuple[str, float]]] = {}

    for query_id in sorted(bm25_queries):
        fused[query_id] = fuse_query(
            bm25_run[query_id],
            dense_run[query_id],
            rrf_k=rrf_k,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    return fused


def write_trec_run(
    fused: dict[str, list[tuple[str, float]]],
    path: Path,
    *,
    run_name: str,
) -> None:
    """Write fused rankings in standard six-column TREC format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as stream:
        for query_id in sorted(fused):
            entries = fused[query_id]

            if not entries:
                raise ValueError(
                    f"Hybrid ranking пуст для query_id={query_id!r}"
                )

            for rank, (doc_id, score) in enumerate(entries, start=1):
                stream.write(
                    f"{query_id} Q0 {doc_id} {rank} {score:.12f} {run_name}\n"
                )

    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Объединить BM25 и Dense TREC run методом "
            "weighted Reciprocal Rank Fusion (RRF)."
        )
    )
    parser.add_argument(
        "--bm25-run",
        type=Path,
        default=DEFAULT_BM25_RUN,
        help=f"BM25 TREC run (по умолчанию: {DEFAULT_BM25_RUN})",
    )
    parser.add_argument(
        "--dense-run",
        type=Path,
        default=DEFAULT_DENSE_RUN,
        help=f"Dense TREC run (по умолчанию: {DEFAULT_DENSE_RUN})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"выходной Hybrid TREC run (по умолчанию: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"константа RRF k (по умолчанию: {DEFAULT_RRF_K})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"число документов в итоговом ranking (по умолчанию: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        help=f"вес BM25 в RRF (по умолчанию: {DEFAULT_BM25_WEIGHT})",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help=f"вес Dense в RRF (по умолчанию: {DEFAULT_DENSE_WEIGHT})",
    )
    parser.add_argument(
        "--run-name",
        default="hybrid-rrf",
        help="имя run в шестой колонке TREC",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rrf_k < 0:
        raise ValueError("--rrf-k должен быть >= 0")
    if args.top_k <= 0:
        raise ValueError("--top-k должен быть > 0")
    if args.bm25_weight < 0:
        raise ValueError("--bm25-weight должен быть >= 0")
    if args.dense_weight < 0:
        raise ValueError("--dense-weight должен быть >= 0")
    if args.bm25_weight == 0 and args.dense_weight == 0:
        raise ValueError("Хотя бы один вес должен быть > 0")

    print(f"BM25 run:      {args.bm25_run}")
    print(f"Dense run:     {args.dense_run}")
    print(f"Output:        {args.output}")
    print(f"RRF k:         {args.rrf_k}")
    print(f"Top-k:         {args.top_k}")
    print(f"BM25 weight:   {args.bm25_weight}")
    print(f"Dense weight:  {args.dense_weight}")
    print()

    bm25_run = parse_trec_run(args.bm25_run)
    dense_run = parse_trec_run(args.dense_run)

    fused = fuse_runs(
        bm25_run,
        dense_run,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

    write_trec_run(
        fused,
        args.output,
        run_name=args.run_name,
    )

    query_count = len(fused)
    result_count = sum(len(entries) for entries in fused.values())

    print("=== Hybrid Weighted RRF ===")
    print(f"Queries:       {query_count:,}")
    print(f"Results:       {result_count:,}")
    print(f"RRF k:         {args.rrf_k}")
    print(f"Top-k:         {args.top_k}")
    print(f"BM25 weight:   {args.bm25_weight}")
    print(f"Dense weight:  {args.dense_weight}")
    print(f"Run saved:     {args.output}")


if __name__ == "__main__":
    main()
