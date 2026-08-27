"""Evaluate a TREC retrieval run with dependency-free IR metrics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from data_utils import (
    SUPPORTED_SPLITS,
    DataFormatError,
    load_qrels,
    load_topics,
    require_file,
    require_python_311,
    validate_referenced_docids,
    validate_split_files,
    write_json,
)
from run_bm25 import default_run_path


METRIC_CUTOFFS = (1, 5, 10)


def default_metrics_path(split: str) -> Path:
    return TABLES_DIR / f"bm25_{split}_metrics.json"


def load_trec_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load and strictly validate a six-column TREC run."""
    require_file(path, "TREC run")
    ranked: dict[str, list[tuple[int, str, float]]] = {}
    run_name: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                raise DataFormatError(
                    f"Ожидалось 6 полей TREC run в {path}:{line_number}, "
                    f"получено {len(parts)}"
                )
            query_id, iteration, doc_id, rank_text, score_text, current_run_name = parts
            if iteration != "Q0":
                raise DataFormatError(
                    f"Второе поле TREC run должно быть Q0 ({path}:{line_number})"
                )
            try:
                rank = int(rank_text)
                score = float(score_text)
            except ValueError as error:
                raise DataFormatError(
                    f"Некорректный rank/score в {path}:{line_number}"
                ) from error
            if rank <= 0 or not math.isfinite(score):
                raise DataFormatError(
                    f"Некорректный rank/score в {path}:{line_number}"
                )
            if run_name is None:
                run_name = current_run_name
            elif current_run_name != run_name:
                raise DataFormatError("В одном TREC run обнаружены разные run_name")
            ranked.setdefault(query_id, []).append((rank, doc_id, score))
    if not ranked:
        raise DataFormatError(f"TREC run не содержит результатов: {path}")

    result: dict[str, list[tuple[str, float]]] = {}
    for query_id, rows in ranked.items():
        rows.sort(key=lambda row: row[0])
        ranks = [rank for rank, _doc_id, _score in rows]
        expected_ranks = list(range(1, len(rows) + 1))
        if ranks != expected_ranks:
            raise DataFormatError(
                f"Ранги query_id={query_id!r} не образуют последовательность 1..N"
            )
        docids = [doc_id for _rank, doc_id, _score in rows]
        if len(docids) != len(set(docids)):
            raise DataFormatError(f"Повтор docid в выдаче query_id={query_id!r}")
        result[query_id] = [
            (doc_id, score) for _rank, doc_id, score in rows
        ]
    return result


def recall_at_k(ranking: list[str], relevances: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, gain in relevances.items() if gain > 0}
    if not relevant:
        raise DataFormatError("Recall не определён для запроса без релевантных документов")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank_at_k(
    ranking: list[str], relevances: dict[str, int], k: int
) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevances.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], relevances: dict[str, int], k: int) -> float:
    def discounted_gain(gain: int, rank: int) -> float:
        return (2.0**gain - 1.0) / math.log2(rank + 1)

    dcg = sum(
        discounted_gain(relevances.get(doc_id, 0), rank)
        for rank, doc_id in enumerate(ranking[:k], start=1)
        if relevances.get(doc_id, 0) > 0
    )
    ideal_gains = sorted(
        (gain for gain in relevances.values() if gain > 0), reverse=True
    )[:k]
    if not ideal_gains:
        raise DataFormatError("nDCG не определён для запроса без релевантных документов")
    ideal_dcg = sum(
        discounted_gain(gain, rank)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / ideal_dcg


def calculate_metrics(
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
) -> dict[str, float]:
    """Compute macro-averaged Recall, reciprocal rank, and nDCG."""
    totals = {f"Recall@{cutoff}": 0.0 for cutoff in METRIC_CUTOFFS}
    totals["MRR@10"] = 0.0
    totals["nDCG@10"] = 0.0
    for query_id in query_ids:
        ranking = [doc_id for doc_id, _score in run[query_id]]
        relevances = qrels[query_id]
        for cutoff in METRIC_CUTOFFS:
            totals[f"Recall@{cutoff}"] += recall_at_k(
                ranking, relevances, cutoff
            )
        totals["MRR@10"] += reciprocal_rank_at_k(ranking, relevances, 10)
        totals["nDCG@10"] += ndcg_at_k(ranking, relevances, 10)
    query_count = len(query_ids)
    return {name: value / query_count for name, value in totals.items()}


def evaluate(
    *,
    split: str,
    run_path: Path,
    docids_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Validate run/qrels consistency, calculate metrics, and save JSON."""
    require_python_311()
    require_file(run_path, "TREC run")
    require_file(docids_path, "docid mapping")

    topics_path, qrels_path = validate_split_files(split)
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    run = load_trec_run(run_path)

    query_ids = [query_id for query_id, _query in topics]
    topic_query_ids = set(query_ids)
    missing_qrels = topic_query_ids - set(qrels)
    missing_run = topic_query_ids - set(run)
    extra_run = set(run) - topic_query_ids

    if missing_qrels:
        raise DataFormatError(
            f"Qrels отсутствуют для {len(missing_qrels):,} запросов split={split}"
        )
    if missing_run or extra_run:
        raise DataFormatError(
            "Query ID в run не совпадают с topics: "
            f"пропущено={len(missing_run):,}, лишних={len(extra_run):,}"
        )

    referenced_docids = {
        doc_id
        for query_id in query_ids
        for doc_id, relevance in qrels[query_id].items()
        if relevance > 0
    }
    referenced_docids.update(
        doc_id for query_rows in run.values() for doc_id, _score in query_rows
    )
    validate_referenced_docids(docids_path, referenced_docids)

    metrics = calculate_metrics(query_ids, qrels, run)
    result: dict[str, Any] = {
        "metrics": metrics,
        "query_count": len(query_ids),
        "run": str(run_path.resolve()),
        "split": split,
    }

    write_json(metrics_path, result)

    print(f"Метрики для split={split} ({len(query_ids):,} запросов):")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.6f}")
    print(f"Метрики сохранены: {metrics_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="dev")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="TREC run для оценки",
    )
    parser.add_argument(
        "--docids",
        type=Path,
        required=True,
        help="docid mapping соответствующего retrieval-индекса",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        required=True,
        help="путь для JSON-файла с рассчитанными метриками",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        split=args.split,
        run_path=args.run,
        docids_path=args.docids,
        metrics_path=args.metrics_output,
    )


if __name__ == "__main__":
    try:
        main()
    except (DataFormatError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        raise SystemExit(1) from error
