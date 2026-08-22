"""Run BM25 retrieval and write a TREC run for a Mr. TyDi split."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from data_utils import (
    BUILD_TIMING_FILENAME,
    DOCIDS_FILENAME,
    INDEX_DIR,
    RUNS_DIR,
    SUPPORTED_SPLITS,
    TABLES_DIR,
    DataFormatError,
    atomic_text_writer,
    create_russian_tokenizer,
    load_bm25_dependencies,
    load_json,
    load_qrels,
    load_topics,
    read_docids,
    require_complete_index,
    require_python_311,
    validate_split_files,
    write_json,
)


RUN_NAME = "bm25"


def default_run_path(split: str) -> Path:
    return RUNS_DIR / f"bm25_{split}.trec"


def default_timing_path(split: str) -> Path:
    return TABLES_DIR / f"bm25_{split}_timing.json"


def _validate_query_ids(
    topics: list[tuple[str, str]], qrels: dict[str, dict[str, int]], split: str
) -> None:
    topic_ids = {query_id for query_id, _query in topics}
    qrel_ids = set(qrels)
    missing = topic_ids - qrel_ids
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise DataFormatError(
            f"Для {len(missing):,} запросов split={split} отсутствуют qrels: {examples}"
        )


def run_search(
    *,
    split: str = "dev",
    top_k: int = 100,
    index_dir: Path = INDEX_DIR,
    run_path: Path | None = None,
    timing_path: Path | None = None,
) -> dict[str, Any]:
    """Load the persisted index, retrieve all split queries, and save the run."""
    require_python_311()
    if top_k <= 0:
        raise ValueError("--top-k должен быть положительным целым числом")
    metadata = require_complete_index(index_dir)
    topics_path, qrels_path = validate_split_files(split)
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    _validate_query_ids(topics, qrels, split)
    bm25s, Stemmer, Tokenizer = load_bm25_dependencies()

    total_start = time.perf_counter()
    index_load_start = time.perf_counter()
    retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=False)
    index_load_seconds = time.perf_counter() - index_load_start
    actual_document_count = int(retriever.scores["num_docs"])
    expected_document_count = int(metadata["document_count"])
    if actual_document_count != expected_document_count:
        raise DataFormatError(
            "Размер загруженного индекса не совпадает с metadata: "
            f"index={actual_document_count:,}, metadata={expected_document_count:,}"
        )
    docids = read_docids(
        index_dir / DOCIDS_FILENAME, expected_count=actual_document_count
    )

    tokenizer = create_russian_tokenizer(Stemmer, Tokenizer)
    tokenizer.load_vocab(index_dir)
    tokenizer.load_stopwords(index_dir)
    queries = [query for _query_id, query in topics]
    query_tokenization_start = time.perf_counter()
    query_tokens = tokenizer.tokenize(
        queries,
        update_vocab=False,
        return_as="ids",
        show_progress=True,
    )
    query_tokenization_seconds = time.perf_counter() - query_tokenization_start

    effective_top_k = min(top_k, actual_document_count)
    retrieval_start = time.perf_counter()
    result_indices, result_scores = retriever.retrieve(
        query_tokens,
        k=effective_top_k,
        show_progress=True,
        n_threads=0,
    )
    query_retrieval_seconds = time.perf_counter() - retrieval_start

    run_path = run_path or default_run_path(split)
    with atomic_text_writer(run_path) as stream:
        for query_position, (query_id, _query) in enumerate(topics):
            seen_docids: set[str] = set()
            for rank, (internal_id_raw, score_raw) in enumerate(
                zip(result_indices[query_position], result_scores[query_position]),
                start=1,
            ):
                internal_id = int(internal_id_raw)
                if not 0 <= internal_id < len(docids):
                    raise DataFormatError(
                        f"bm25s вернул внутренний docid вне mapping: {internal_id}"
                    )
                doc_id = docids[internal_id]
                if doc_id in seen_docids:
                    raise DataFormatError(
                        f"Повтор docid={doc_id!r} в выдаче query_id={query_id!r}"
                    )
                seen_docids.add(doc_id)
                score = float(score_raw)
                if not math.isfinite(score):
                    raise DataFormatError(
                        f"Некорректный score для query_id={query_id!r}: {score}"
                    )
                stream.write(
                    f"{query_id} Q0 {doc_id} {rank} {score:.10g} {RUN_NAME}\n"
                )

    build_timing = load_json(
        index_dir / BUILD_TIMING_FILENAME, description="статистика построения индекса"
    )
    query_count = len(topics)
    query_processing_seconds = query_tokenization_seconds + query_retrieval_seconds
    timing: dict[str, Any] = {
        **build_timing,
        "average_query_processing_seconds": query_processing_seconds / query_count,
        "average_query_seconds": query_retrieval_seconds / query_count,
        "effective_top_k": effective_top_k,
        "experiment_total_seconds": time.perf_counter() - total_start,
        "index_load_seconds": index_load_seconds,
        "query_count": query_count,
        "query_processing_seconds": query_processing_seconds,
        "query_retrieval_seconds": query_retrieval_seconds,
        "query_tokenization_seconds": query_tokenization_seconds,
        "requested_top_k": top_k,
        "split": split,
    }
    timing_path = timing_path or default_timing_path(split)
    write_json(timing_path, timing)

    print(f"TREC run сохранён: {run_path}")
    print(f"Запросов:             {query_count:,}")
    print(f"Top-k:                {effective_top_k}")
    print(f"Токенизация запросов: {query_tokenization_seconds:.4f} с")
    print(f"Поиск:                {query_retrieval_seconds:.4f} с")
    print(f"Среднее на запрос:    {timing['average_query_seconds']:.6f} с")
    print(f"Timing сохранён:      {timing_path}")
    return timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="dev")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_search(split=args.split, top_k=args.top_k)


if __name__ == "__main__":
    try:
        main()
    except (DataFormatError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        raise SystemExit(1) from error
