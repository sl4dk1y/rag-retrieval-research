"""Build a compact IVF+SQ8 FAISS index with multilingual-e5-small.

The full run is intentionally long on Apple M4 (~one day from the benchmark),
so a smoke-test mode is provided. The final index uses IVF for practical search
speed and 8-bit scalar quantization to keep the 9.6M-vector index manageable.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dense_utils import (
    DEFAULT_BATCH_SIZE,
    DENSE_INDEX_DIR,
    FULL_CORPUS_DOCUMENTS,
    MODEL_NAME,
    batched_corpus,
    iter_corpus,
    write_json,
)


INDEX_FILENAME = "index.faiss"
DOCIDS_FILENAME = "docids.txt"
METADATA_FILENAME = "metadata.json"
TIMING_FILENAME = "build_timing.json"
SUCCESS_FILENAME = ".complete"

DEFAULT_TRAIN_DOCS = 100_000
DEFAULT_NLIST = 4096


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS запрошен, но недоступен")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--train-docs", type=int, default=DEFAULT_TRAIN_DOCS)
    parser.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Subset для smoke-test; такой индекс нельзя использовать как финальный.",
    )
    return parser.parse_args()


def complete(index_dir: Path) -> bool:
    return all(
        (index_dir / name).is_file()
        for name in (
            INDEX_FILENAME,
            DOCIDS_FILENAME,
            METADATA_FILENAME,
            TIMING_FILENAME,
            SUCCESS_FILENAME,
        )
    )


def encode_training_sample(
    model: SentenceTransformer,
    *,
    train_docs: int,
    batch_size: int,
    dimension: int,
) -> tuple[np.ndarray, float]:
    """Encode a deterministic prefix sample used only to train IVF/SQ8."""
    vectors = np.empty((train_docs, dimension), dtype=np.float32)
    written = 0
    encode_seconds = 0.0
    progress = tqdm(total=train_docs, unit="doc", desc="FAISS training sample")

    for _docids, texts in batched_corpus(iter_corpus(), batch_size):
        remaining = train_docs - written
        if remaining <= 0:
            break
        texts = texts[:remaining]
        start = time.perf_counter()
        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        encode_seconds += time.perf_counter() - start
        batch_vectors = np.asarray(embeddings, dtype=np.float32)
        vectors[written : written + len(texts)] = batch_vectors
        written += len(texts)
        progress.update(len(texts))
        if written >= train_docs:
            break

    progress.close()
    if written != train_docs:
        raise RuntimeError(
            f"Для обучения FAISS получено {written:,} документов вместо {train_docs:,}"
        )
    return vectors, encode_seconds


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.train_docs <= 0 or args.nlist <= 0:
        raise ValueError("batch-size/train-docs/nlist должны быть > 0")
    if args.max_docs is not None and args.max_docs <= 0:
        raise ValueError("--max-docs должен быть > 0")

    target_docs = args.max_docs or FULL_CORPUS_DOCUMENTS
    train_docs = min(args.train_docs, target_docs)
    if train_docs < args.nlist * 10:
        raise ValueError(
            "Слишком мало train-docs для выбранного nlist. "
            "Используйте минимум примерно 10 * nlist."
        )

    device = choose_device(args.device)
    index_dir = (
        DENSE_INDEX_DIR
        if args.max_docs is None
        else DENSE_INDEX_DIR.parent / f"dense-e5-small-smoke-{args.max_docs}"
    )

    if complete(index_dir) and not args.force:
        print(f"Готовый dense-индекс уже существует: {index_dir}")
        return
    if index_dir.exists() and not args.force:
        raise RuntimeError(
            f"Каталог {index_dir} уже существует, но индекс незавершён. "
            "Проверьте его и используйте --force."
        )

    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"Documents:   {target_docs:,}")
    print(f"Batch size:  {args.batch_size}")
    print(f"IVF nlist:   {args.nlist}")
    print(f"Train docs:  {train_docs:,}")
    print(f"Index dir:   {index_dir}")

    total_start = time.perf_counter()
    model_start = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    model_load_seconds = time.perf_counter() - model_start
    dimension = model.get_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Не удалось определить embedding dimension")

    training_vectors, training_encode_seconds = encode_training_sample(
        model,
        train_docs=train_docs,
        batch_size=args.batch_size,
        dimension=int(dimension),
    )

    quantizer = faiss.IndexFlatIP(int(dimension))
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        int(dimension),
        args.nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )
    train_start = time.perf_counter()
    index.train(training_vectors)
    faiss_train_seconds = time.perf_counter() - train_start
    del training_vectors

    if not index.is_trained:
        raise RuntimeError("FAISS index не перешёл в trained state")

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent)
    )
    processed = 0
    encoding_seconds = 0.0
    add_seconds = 0.0
    progress = tqdm(total=target_docs, unit="doc", desc="Dense index")

    try:
        with (temp_dir / DOCIDS_FILENAME).open("w", encoding="utf-8") as mapping:
            for docids, texts in batched_corpus(iter_corpus(), args.batch_size):
                remaining = target_docs - processed
                if remaining <= 0:
                    break
                if len(texts) > remaining:
                    docids, texts = docids[:remaining], texts[:remaining]

                encode_start = time.perf_counter()
                embeddings = model.encode(
                    texts,
                    batch_size=len(texts),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                encoding_seconds += time.perf_counter() - encode_start
                vectors = np.asarray(embeddings, dtype=np.float32)

                add_start = time.perf_counter()
                index.add(vectors)
                add_seconds += time.perf_counter() - add_start

                for docid in docids:
                    mapping.write(docid + "\n")

                processed += len(docids)
                progress.update(len(docids))
                del embeddings, vectors

                if processed >= target_docs:
                    break

        progress.close()
        if processed != target_docs or index.ntotal != target_docs:
            raise RuntimeError(
                f"Неполный индекс: processed={processed:,}, ntotal={index.ntotal:,}, "
                f"expected={target_docs:,}"
            )

        save_start = time.perf_counter()
        faiss.write_index(index, str(temp_dir / INDEX_FILENAME))
        save_seconds = time.perf_counter() - save_start

        timing = {
            "model_load_seconds": model_load_seconds,
            "training_sample_encode_seconds": training_encode_seconds,
            "faiss_train_seconds": faiss_train_seconds,
            "corpus_encoding_seconds": encoding_seconds,
            "faiss_add_seconds": add_seconds,
            "index_save_seconds": save_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "documents_per_second": processed / encoding_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        metadata = {
            "model": args.model,
            "device": device,
            "batch_size": args.batch_size,
            "embedding_dimension": int(dimension),
            "normalized_embeddings": True,
            "document_prefix": "passage: ",
            "document_count": processed,
            "is_subset": args.max_docs is not None,
            "max_docs": args.max_docs,
            "faiss_type": "IVFScalarQuantizer(SQ8, inner-product)",
            "nlist": args.nlist,
            "train_docs": train_docs,
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "torch_version": torch.__version__,
        }
        write_json(temp_dir / TIMING_FILENAME, timing)
        write_json(temp_dir / METADATA_FILENAME, metadata)
        (temp_dir / SUCCESS_FILENAME).write_text("ok\n", encoding="utf-8")

        if index_dir.exists():
            shutil.rmtree(index_dir)
        os.replace(temp_dir, index_dir)
    except BaseException:
        progress.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print("\n=== Dense index build ===")
    print(f"Documents:      {processed:,}")
    print(f"Dimension:      {dimension}")
    print(f"Encoding:       {encoding_seconds / 3600:.2f} h")
    print(f"Documents/sec:  {processed / encoding_seconds:.2f}")
    print(f"FAISS add:      {add_seconds:.2f} s")
    print(f"Index save:     {save_seconds:.2f} s")
    print(f"Peak RAM:       {peak_rss_bytes() / 1024**3:.2f} GiB")
    print(f"Index saved:    {index_dir}")


if __name__ == "__main__":
    main()
