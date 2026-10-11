"""Run multilingual-E5 Dense Retrieval and write a standard TREC run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from dense_utils import (
    DEFAULT_TOP_K,
    DENSE_INDEX_DIR,
    MODEL_NAME,
    RUNS_DIR,
    TABLES_DIR,
    load_topics,
    write_json,
)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS запрошен, но недоступен")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DENSE_INDEX_DIR,
        help=f"каталог FAISS-индекса (по умолчанию: {DENSE_INDEX_DIR})",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        default=None,
        help="точный путь для TREC run; по умолчанию используется исторический путь",
    )
    parser.add_argument(
        "--timing-path",
        type=Path,
        default=None,
        help="точный путь для JSON timing; по умолчанию используется исторический путь",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.nprobe <= 0:
        raise ValueError("top-k и nprobe должны быть > 0")

    for name in ("index.faiss", "docids.txt", "metadata.json", ".complete"):
        if not (args.index_dir / name).is_file():
            raise FileNotFoundError(
                f"Dense index незавершён: отсутствует {args.index_dir / name}"
            )

    metadata = json.loads(
        (args.index_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("is_subset"):
        raise RuntimeError("Subset dense index нельзя использовать для финальной оценки")
    if metadata.get("model") != args.model:
        raise RuntimeError(
            f"Индекс построен {metadata.get('model')!r}, запрошена {args.model!r}"
        )

    print("Загрузка FAISS index...")
    load_start = time.perf_counter()
    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    index.nprobe = args.nprobe
    index_load_seconds = time.perf_counter() - load_start

    print("Загрузка docid mapping...")
    docids = (args.index_dir / "docids.txt").read_text(encoding="utf-8").splitlines()
    if len(docids) != index.ntotal:
        raise RuntimeError(
            f"docids={len(docids):,}, index.ntotal={index.ntotal:,}"
        )

    topics = load_topics(args.split)
    device = choose_device(args.device)
    print(f"Загрузка модели {args.model} на {device}...")
    model_start = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    model_load_seconds = time.perf_counter() - model_start

    query_ids = [query_id for query_id, _ in topics]
    query_texts = [text for _, text in topics]
    encode_start = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_encode_seconds = time.perf_counter() - encode_start
    query_vectors = np.asarray(query_embeddings, dtype=np.float32)

    search_start = time.perf_counter()
    scores, positions = index.search(query_vectors, args.top_k)
    search_seconds = time.perf_counter() - search_start

    run_path = args.run_path or RUNS_DIR / f"dense_e5_small_{args.split}.trec"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as stream:
        for query_index, query_id in enumerate(query_ids):
            for rank, (position, score) in enumerate(
                zip(positions[query_index], scores[query_index], strict=True), start=1
            ):
                if position < 0:
                    continue
                docid = docids[int(position)]
                stream.write(
                    f"{query_id} Q0 {docid} {rank} {float(score):.8f} dense-e5-small\n"
                )

    timing = {
        "split": args.split,
        "query_count": len(topics),
        "top_k": args.top_k,
        "nprobe": args.nprobe,
        "model": args.model,
        "device": device,
        "index_load_seconds": index_load_seconds,
        "model_load_seconds": model_load_seconds,
        "query_encode_seconds": query_encode_seconds,
        "search_seconds": search_seconds,
        "average_search_seconds_per_query": search_seconds / len(topics),
    }
    timing_path = args.timing_path or TABLES_DIR / f"dense_e5_small_{args.split}_timing.json"
    write_json(timing_path, timing)

    print("\n=== Dense retrieval ===")
    print(f"Queries:        {len(topics):,}")
    print(f"Top-k:          {args.top_k}")
    print(f"nprobe:         {args.nprobe}")
    print(f"Query encoding: {query_encode_seconds:.2f} s")
    print(f"FAISS search:   {search_seconds:.2f} s")
    print(f"Run saved:      {run_path}")
    print(f"Timing saved:   {timing_path}")


if __name__ == "__main__":
    main()
