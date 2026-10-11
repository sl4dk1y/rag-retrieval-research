"""Run BGE-M3 dense retrieval against the frozen reduced Mr. TyDi index."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from bge_m3_utils import (
    BGE_INDEX_DIR,
    DEFAULT_NPROBE,
    DEFAULT_RUN_PATH,
    DEFAULT_TIMING_PATH,
    DEFAULT_TOP_K,
    DOCIDS_FILENAME,
    EMBEDDING_DIMENSION,
    INDEX_FILENAME,
    MAX_SEQUENCE_LENGTH,
    METADATA_FILENAME,
    MODEL_NAME,
    choose_device,
    query_text,
    synchronize_device,
    timed_encode,
    validate_bge_metadata,
    validate_index_artifacts,
)
from data_utils import load_topics, split_paths, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--nprobe",
        type=int,
        default=DEFAULT_NPROBE,
        help=(
            "число IVF lists при поиске; явный engineering-параметр, "
            "по умолчанию используется значение controlled reduced experiment"
        ),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--run-path", type=Path, default=DEFAULT_RUN_PATH)
    parser.add_argument("--timing-path", type=Path, default=DEFAULT_TIMING_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.nprobe <= 0:
        raise ValueError("top-k и nprobe должны быть > 0")
    metadata = validate_index_artifacts(args.index_dir)
    validate_bge_metadata(metadata)

    print("Загрузка BGE-M3 FAISS index...")
    index_load_start = time.perf_counter()
    index = faiss.read_index(str(args.index_dir / INDEX_FILENAME))
    validate_index_artifacts(args.index_dir, index_ntotal=index.ntotal)
    index.nprobe = args.nprobe
    index_load_seconds = time.perf_counter() - index_load_start
    docids = (args.index_dir / DOCIDS_FILENAME).read_text(encoding="utf-8").splitlines()
    if len(docids) != index.ntotal:
        raise RuntimeError(f"docids={len(docids):,}, index.ntotal={index.ntotal:,}")
    if len(docids) != metadata["document_count"]:
        raise RuntimeError(
            f"docids={len(docids):,}, metadata document_count={metadata['document_count']:,}"
        )

    topics_path, _qrels_path = split_paths(args.split)
    topics = load_topics(topics_path)
    device = choose_device(args.device)
    print(f"Загрузка модели {MODEL_NAME} на {device}...")
    model_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device)
    model_load_seconds = time.perf_counter() - model_start
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    if model.get_embedding_dimension() != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"BGE-M3 embedding dimension={model.get_embedding_dimension()!r}; "
            f"ожидалось {EMBEDDING_DIMENSION}"
        )
    query_ids = [query_id for query_id, _text in topics]
    query_texts = [query_text(text) for _query_id, text in topics]
    query_embeddings, query_encode_seconds = timed_encode(
        model, query_texts, batch_size=16, device=device
    )
    query_vectors = np.asarray(query_embeddings, dtype=np.float32)
    if query_vectors.shape != (len(query_texts), EMBEDDING_DIMENSION):
        raise RuntimeError(
            "BGE-M3 вернула query embedding неожиданной формы: "
            f"{query_vectors.shape}"
        )
    synchronize_device(device)
    search_start = time.perf_counter()
    scores, positions = index.search(query_vectors, args.top_k)
    synchronize_device(device)
    search_seconds = time.perf_counter() - search_start

    args.run_path.parent.mkdir(parents=True, exist_ok=True)
    with args.run_path.open("w", encoding="utf-8") as stream:
        for query_index, query_id in enumerate(query_ids):
            for rank, (position, score) in enumerate(
                zip(positions[query_index], scores[query_index], strict=True), start=1
            ):
                if position < 0:
                    continue
                stream.write(
                    f"{query_id} Q0 {docids[int(position)]} {rank} "
                    f"{float(score):.8f} dense-bge-m3\n"
                )
    timing = {
        "split": args.split,
        "query_count": len(topics),
        "top_k": args.top_k,
        "nprobe": args.nprobe,
        "model": MODEL_NAME,
        "device": device,
        "index_load_seconds": index_load_seconds,
        "model_load_seconds": model_load_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": search_seconds,
        "average_search_seconds_per_query": search_seconds / len(topics),
    }
    write_json(args.timing_path, timing)
    print("\n=== BGE-M3 dense retrieval ===")
    print(f"Queries:        {len(topics):,}")
    print(f"Top-k:          {args.top_k}")
    print(f"nprobe:         {args.nprobe}")
    print(f"Query encoding: {query_encode_seconds:.2f} s")
    print(f"FAISS search:   {search_seconds:.2f} s")
    print(f"Run saved:      {args.run_path}")
    print(f"Timing saved:   {args.timing_path}")


if __name__ == "__main__":
    main()
