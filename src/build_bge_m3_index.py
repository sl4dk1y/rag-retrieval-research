"""Build the isolated BGE-M3 dense index for frozen reduced Mr. TyDi Russian."""

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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bge_m3_utils import (
    BGE_INDEX_DIR,
    DEFAULT_NLIST,
    DEFAULT_TRAIN_DOCS,
    DOCIDS_FILENAME,
    EMBEDDING_DIMENSION,
    INDEX_BATCH_SIZE,
    INDEX_FILENAME,
    MAX_SEQUENCE_LENGTH,
    METADATA_FILENAME,
    MODEL_NAME,
    REDUCED_CORPUS_PATH,
    SUCCESS_FILENAME,
    TIMING_FILENAME,
    batched_corpus,
    choose_device,
    index_is_complete,
    iter_bge_corpus,
    runtime_metadata,
    timed_encode,
    validate_index_artifacts,
    verify_corpus_sha256,
)
from data_utils import write_json


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", type=Path, default=REDUCED_CORPUS_PATH)
    parser.add_argument("--index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--expected-document-count", type=int, default=1_000_000)
    parser.add_argument("--corpus-sha256", default=None)
    parser.add_argument("--batch-size", type=int, default=INDEX_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--nlist",
        type=int,
        default=DEFAULT_NLIST,
        help=(
            "число IVF lists; явный engineering-параметр, по умолчанию "
            "используется значение controlled reduced experiment"
        ),
    )
    parser.add_argument("--train-docs", type=int, default=DEFAULT_TRAIN_DOCS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def encode_training_sample(
    model: SentenceTransformer,
    *,
    corpus_path: Path,
    train_docs: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """Encode a deterministic corpus prefix for IVF/SQ8 training."""
    vectors = np.empty((train_docs, EMBEDDING_DIMENSION), dtype=np.float32)
    written = 0
    encode_seconds = 0.0
    progress = tqdm(total=train_docs, unit="doc", desc="BGE-M3 FAISS training sample")
    try:
        for _docids, texts in batched_corpus(iter_bge_corpus(corpus_path), batch_size):
            remaining = train_docs - written
            if remaining <= 0:
                break
            texts = texts[:remaining]
            embeddings, elapsed = timed_encode(
                model, texts, batch_size=len(texts), device=device
            )
            batch_vectors = np.asarray(embeddings, dtype=np.float32)
            if batch_vectors.shape != (len(texts), EMBEDDING_DIMENSION):
                raise RuntimeError(
                    "BGE-M3 вернула embedding неожиданной формы: "
                    f"{batch_vectors.shape}, ожидалось ({len(texts)}, {EMBEDDING_DIMENSION})"
                )
            vectors[written : written + len(texts)] = batch_vectors
            written += len(texts)
            encode_seconds += elapsed
            progress.update(len(texts))
    finally:
        progress.close()
    if written != train_docs:
        raise RuntimeError(
            f"Для обучения FAISS получено {written:,} документов вместо {train_docs:,}"
        )
    return vectors, encode_seconds


def main() -> None:
    args = parse_args()
    if min(
        args.expected_document_count, args.batch_size, args.nlist, args.train_docs
    ) <= 0:
        raise ValueError("expected-document-count/batch-size/nlist/train-docs должны быть > 0")
    if not args.corpus_path.is_file():
        raise FileNotFoundError(f"Не найден корпус: {args.corpus_path}")

    total_start = time.perf_counter()
    hash_start = time.perf_counter()
    verified_corpus_sha256 = verify_corpus_sha256(
        args.corpus_path, args.corpus_sha256
    )
    hash_seconds = time.perf_counter() - hash_start

    train_docs = min(args.train_docs, args.expected_document_count)
    if train_docs < args.nlist * 10:
        raise ValueError(
            "Слишком мало train-docs для выбранного nlist. "
            "Используйте минимум примерно 10 * nlist."
        )
    device = choose_device(args.device)
    if index_is_complete(args.index_dir) and not args.force:
        print(f"Готовый BGE-M3 индекс уже существует: {args.index_dir}")
        return
    if args.index_dir.exists() and not args.force:
        raise RuntimeError(
            f"Каталог {args.index_dir} уже существует, но индекс незавершён. "
            "Проверьте его и используйте --force."
        )

    print(f"Model:       {MODEL_NAME}")
    print(f"Device:      {device}")
    print(f"Documents:   {args.expected_document_count:,}")
    print(f"Batch size:  {args.batch_size}")
    print(f"IVF nlist:   {args.nlist}")
    print(f"Train docs:  {train_docs:,}")
    print(f"Corpus:      {args.corpus_path}")
    print(f"Index dir:   {args.index_dir}")

    model_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device=device)
    model_load_seconds = time.perf_counter() - model_start
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    dimension = model.get_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"BGE-M3 embedding dimension={dimension!r}; ожидалось {EMBEDDING_DIMENSION}"
        )

    training_vectors, training_encode_seconds = encode_training_sample(
        model,
        corpus_path=args.corpus_path,
        train_docs=train_docs,
        batch_size=args.batch_size,
        device=device,
    )
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        EMBEDDING_DIMENSION,
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

    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.index_dir.name}-", dir=args.index_dir.parent)
    )
    processed = 0
    encoding_seconds = 0.0
    add_seconds = 0.0
    seen_docids: set[str] = set()
    progress = tqdm(
        total=args.expected_document_count, unit="doc", desc="BGE-M3 dense index"
    )
    try:
        with (temp_dir / DOCIDS_FILENAME).open("w", encoding="utf-8") as mapping:
            for docids, texts in batched_corpus(
                iter_bge_corpus(args.corpus_path), args.batch_size
            ):
                if processed + len(docids) > args.expected_document_count:
                    raise RuntimeError(
                        "Корпус содержит больше документов, чем ожидалось: "
                        f"expected={args.expected_document_count:,}, "
                        f"encountered_at_least={processed + len(docids):,}"
                    )
                batch_docids: set[str] = set()
                for docid in docids:
                    if docid in seen_docids or docid in batch_docids:
                        raise RuntimeError(f"Повтор docid в BGE-M3 корпусе: {docid!r}")
                    batch_docids.add(docid)
                seen_docids.update(batch_docids)

                embeddings, elapsed = timed_encode(
                    model, texts, batch_size=len(texts), device=device
                )
                vectors = np.asarray(embeddings, dtype=np.float32)
                if vectors.shape != (len(texts), EMBEDDING_DIMENSION):
                    raise RuntimeError(
                        "BGE-M3 вернула embedding неожиданной формы: "
                        f"{vectors.shape}, ожидалось ({len(texts)}, {EMBEDDING_DIMENSION})"
                    )
                encoding_seconds += elapsed
                add_start = time.perf_counter()
                index.add(vectors)
                add_seconds += time.perf_counter() - add_start
                for docid in docids:
                    mapping.write(docid + "\n")
                processed += len(docids)
                progress.update(len(docids))
                del embeddings, vectors
            mapping.flush()
            os.fsync(mapping.fileno())
        progress.close()
        if processed != args.expected_document_count or index.ntotal != processed:
            raise RuntimeError(
                f"Неполный индекс: processed={processed:,}, ntotal={index.ntotal:,}, "
                f"expected={args.expected_document_count:,}"
            )
        save_start = time.perf_counter()
        faiss.write_index(index, str(temp_dir / INDEX_FILENAME))
        save_seconds = time.perf_counter() - save_start
        timing = {
            "corpus_sha256_seconds": hash_seconds,
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
            "model": MODEL_NAME,
            "corpus_path": str(args.corpus_path.resolve()),
            "corpus_sha256": verified_corpus_sha256,
            "expected_document_count": args.expected_document_count,
            "document_count": processed,
            "device": device,
            "batch_size": args.batch_size,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "normalized_embeddings": True,
            "document_format": "title + ' ' + text",
            "query_format": "raw query text",
            "retrieval_mode": "dense",
            "faiss_type": "IVFScalarQuantizer(SQ8, inner-product)",
            "metric": "inner_product",
            "nlist": args.nlist,
            "train_docs": train_docs,
            "index_file_size_bytes": (temp_dir / INDEX_FILENAME).stat().st_size,
            **runtime_metadata(),
        }
        write_json(temp_dir / TIMING_FILENAME, timing)
        write_json(temp_dir / METADATA_FILENAME, metadata)
        with (temp_dir / SUCCESS_FILENAME).open("w", encoding="utf-8") as marker:
            marker.write("ok\n")
            marker.flush()
            os.fsync(marker.fileno())
        validate_index_artifacts(temp_dir, index_ntotal=index.ntotal)
        if args.index_dir.exists():
            if not args.force:
                raise FileExistsError(
                    f"Каталог индекса появился во время построения: {args.index_dir}"
                )
            shutil.rmtree(args.index_dir)
        os.replace(temp_dir, args.index_dir)
    except BaseException:
        progress.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print("\n=== BGE-M3 dense index build ===")
    print(f"Documents:      {processed:,}")
    print(f"Dimension:      {EMBEDDING_DIMENSION}")
    print(f"Encoding:       {encoding_seconds / 3600:.2f} h")
    print(f"Documents/sec:  {processed / encoding_seconds:.2f}")
    print(f"FAISS add:      {add_seconds:.2f} s")
    print(f"Index save:     {save_seconds:.2f} s")
    print(f"Peak RAM:       {peak_rss_bytes() / 1024**3:.2f} GiB")
    print(f"Index saved:    {args.index_dir}")


if __name__ == "__main__":
    main()
