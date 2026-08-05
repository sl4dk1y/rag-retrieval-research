"""Build (if needed), retrieve, evaluate, and summarize the BM25 experiment."""

from __future__ import annotations

import argparse
import sys
import time

from build_bm25_index import build_index
from data_utils import (
    INDEX_DIR,
    SUPPORTED_SPLITS,
    DataFormatError,
    index_is_complete,
    require_complete_index,
    require_python_311,
)
from evaluate_run import evaluate
from run_bm25 import run_search


def run_experiment(*, split: str = "dev", top_k: int = 100) -> None:
    """Execute the complete lexical baseline workflow."""
    require_python_311()
    experiment_start = time.perf_counter()
    built_now = not index_is_complete(INDEX_DIR)
    if not built_now:
        try:
            require_complete_index(INDEX_DIR)
        except DataFormatError:
            built_now = True
    if built_now:
        replace_incomplete = INDEX_DIR.exists()
        if replace_incomplete:
            print("Найден незавершённый индекс — он будет заменён после построения.")
        else:
            print("Готовый индекс не найден — начинаю построение.")
        build_index(force=replace_incomplete)
    else:
        print(f"Используется существующий индекс: {INDEX_DIR}")

    timing = run_search(split=split, top_k=top_k)
    evaluation = evaluate(split=split)
    elapsed = time.perf_counter() - experiment_start

    print("\nИтоговая сводка")
    print(f"  Split:                 {split}")
    print(f"  Индекс построен сейчас: {'да' if built_now else 'нет'}")
    print(f"  Документов:            {timing['document_count']:,}")
    print(f"  Запросов:              {timing['query_count']:,}")
    print(f"  Top-k:                 {timing['effective_top_k']}")
    for name, value in evaluation["metrics"].items():
        print(f"  {name:<22} {value:.6f}")
    print(f"  Поиск:                 {timing['query_retrieval_seconds']:.4f} с")
    print(f"  Среднее на запрос:     {timing['average_query_seconds']:.6f} с")
    print(f"  Текущее выполнение:    {elapsed:.2f} с")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="dev")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(split=args.split, top_k=args.top_k)


if __name__ == "__main__":
    try:
        main()
    except (DataFormatError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        raise SystemExit(1) from error
