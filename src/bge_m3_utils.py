"""Shared, BGE-M3-specific helpers for the frozen reduced-corpus experiment."""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from data_utils import DataFormatError, PROJECT_ROOT, iter_jsonl_gz, load_json


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
MAX_SEQUENCE_LENGTH = 512
INDEX_BATCH_SIZE = 2
DEFAULT_TOP_K = 100

REDUCED_CORPUS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "mr-tydi-russian-reduced-1m" / "corpus.jsonl.gz"
)
BGE_INDEX_DIR = (
    PROJECT_ROOT / "indexes" / "mr-tydi-russian-reduced-1m" / "dense-bge-m3"
)
BGE_RUNS_DIR = PROJECT_ROOT / "results" / "reduced-1m" / "runs"
BGE_TABLES_DIR = PROJECT_ROOT / "results" / "reduced-1m" / "tables"
DEFAULT_RUN_PATH = BGE_RUNS_DIR / "dense_bge_m3_dev.trec"
DEFAULT_TIMING_PATH = BGE_TABLES_DIR / "dense_bge_m3_dev_timing.json"

INDEX_FILENAME = "index.faiss"
DOCIDS_FILENAME = "docids.txt"
METADATA_FILENAME = "metadata.json"
TIMING_FILENAME = "build_timing.json"
SUCCESS_FILENAME = ".complete"

# These values are inherited from the controlled reduced-corpus IVF experiment.
# They are engineering defaults, not selected from BGE-M3 test results, and remain
# explicit CLI parameters in both the builder and retrieval scripts.
DEFAULT_NLIST = 2048
DEFAULT_NPROBE = 256
DEFAULT_TRAIN_DOCS = 100_000

SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def document_text(title: str, text: str) -> str:
    """Return the frozen BGE-M3 document representation without a prefix."""
    return f"{title} {text}"


def query_text(text: str) -> str:
    """Return raw BGE-M3 query text without a prefix or instruction."""
    return text


def iter_bge_corpus(path: Path) -> Iterator[tuple[str, str]]:
    """Yield validated Mr. TyDi records as BGE-M3 document text."""
    for record_number, record in enumerate(
        iter_jsonl_gz(path, description="Чтение BGE-M3 корпуса"), start=1
    ):
        try:
            docid = record["docid"]
            title = record["title"]
            text = record["text"]
        except KeyError as error:
            raise DataFormatError(
                f"В записи корпуса {record_number:,} отсутствует поле {error.args[0]!r}"
            ) from error
        if not isinstance(docid, str) or not docid:
            raise DataFormatError(
                f"Некорректный docid в записи корпуса {record_number:,}: {docid!r}"
            )
        if not isinstance(title, str) or not isinstance(text, str):
            raise DataFormatError(
                f"title/text должны быть строками в записи корпуса {record_number:,}"
            )
        yield docid, document_text(title, text)


def batched_corpus(
    iterator: Iterator[tuple[str, str]], batch_size: int
) -> Iterator[tuple[list[str], list[str]]]:
    """Batch corpus records while retaining only one encoder batch in memory."""
    docids: list[str] = []
    texts: list[str] = []
    for docid, text in iterator:
        docids.append(docid)
        texts.append(text)
        if len(texts) == batch_size:
            yield docids, texts
            docids, texts = [], []
    if texts:
        yield docids, texts


def choose_device(requested: str) -> str:
    """Resolve the frozen MPS-preferred device policy."""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS запрошен, но недоступен")
    return requested


def synchronize_device(device: str) -> None:
    """Synchronize MPS before a timing boundary; CPU operations are synchronous."""
    if device == "mps":
        torch.mps.synchronize()


def timed_encode(model: Any, texts: list[str], *, batch_size: int, device: str) -> tuple[Any, float]:
    """Encode normalized dense vectors with MPS-aware elapsed time."""
    synchronize_device(device)
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    synchronize_device(device)
    return embeddings, time.perf_counter() - start


def verify_corpus_sha256(corpus_path: Path, expected_digest: str | None) -> str | None:
    """Verify an optional SHA-256 digest without loading the corpus into memory."""
    if expected_digest is None:
        return None
    if SHA256_PATTERN.fullmatch(expected_digest) is None:
        raise ValueError("--corpus-sha256 должен содержать ровно 64 hex-символа")

    digest = hashlib.sha256()
    with corpus_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual_digest = digest.hexdigest()
    if actual_digest.casefold() != expected_digest.casefold():
        raise ValueError(
            "SHA-256 корпуса не совпадает: "
            f"expected={expected_digest.casefold()}, actual={actual_digest}"
        )
    return actual_digest


def validate_index_artifacts(
    index_dir: Path, *, index_ntotal: int | None = None
) -> dict[str, Any]:
    """Validate BGE-M3 artifacts without loading a model or encoding documents."""
    required = (
        INDEX_FILENAME,
        DOCIDS_FILENAME,
        METADATA_FILENAME,
        TIMING_FILENAME,
        SUCCESS_FILENAME,
    )
    for name in required:
        path = index_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Отсутствует BGE-M3 artifact: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Пустой BGE-M3 artifact: {path}")

    metadata = load_json(index_dir / METADATA_FILENAME, description="BGE-M3 metadata")
    document_count = metadata.get("document_count")
    expected_document_count = metadata.get("expected_document_count")
    if not isinstance(document_count, int) or document_count <= 0:
        raise RuntimeError(f"Некорректный BGE-M3 metadata document_count={document_count!r}")
    if not isinstance(expected_document_count, int) or expected_document_count <= 0:
        raise RuntimeError(
            "Некорректный BGE-M3 metadata "
            f"expected_document_count={expected_document_count!r}"
        )
    if document_count != expected_document_count:
        raise RuntimeError(
            "BGE-M3 metadata document_count не совпадает с expected_document_count: "
            f"{document_count:,} != {expected_document_count:,}"
        )

    docid_count = 0
    with (index_dir / DOCIDS_FILENAME).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.rstrip("\n\r"):
                raise RuntimeError(
                    f"Пустой docid в {index_dir / DOCIDS_FILENAME}:{line_number}"
                )
            docid_count += 1
    if docid_count != document_count:
        raise RuntimeError(
            "Число docid не совпадает с BGE-M3 metadata document_count: "
            f"docids={docid_count:,}, metadata={document_count:,}"
        )
    if index_ntotal is not None and index_ntotal != document_count:
        raise RuntimeError(
            "FAISS ntotal не совпадает с BGE-M3 metadata document_count: "
            f"ntotal={index_ntotal:,}, metadata={document_count:,}"
        )
    return metadata


def index_is_complete(index_dir: Path) -> bool:
    """Return whether BGE-M3 artifacts and the FAISS/document mapping agree."""
    try:
        import faiss

        index = faiss.read_index(str(index_dir / INDEX_FILENAME))
        validate_index_artifacts(index_dir, index_ntotal=index.ntotal)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return True


def validate_bge_metadata(metadata: dict[str, Any]) -> None:
    """Reject an index whose immutable BGE-M3 configuration is incompatible."""
    required = {
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalized_embeddings": True,
        "document_format": "title + ' ' + text",
        "query_format": "raw query text",
        "retrieval_mode": "dense",
        "metric": "inner_product",
        "faiss_type": "IVFScalarQuantizer(SQ8, inner-product)",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Несовместимый BGE-M3 metadata: {key}={metadata.get(key)!r}, "
                f"ожидалось {expected!r}"
            )
    for key in ("document_count", "expected_document_count", "nlist", "train_docs"):
        if not isinstance(metadata.get(key), int) or metadata[key] <= 0:
            raise RuntimeError(f"Некорректный BGE-M3 metadata: {key}={metadata.get(key)!r}")


def runtime_metadata() -> dict[str, str]:
    """Return inexpensive runtime version data useful for reproducibility."""
    import faiss
    import numpy
    import sentence_transformers

    return {
        "python_version": sys.version.split()[0],
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "numpy_version": numpy.__version__,
        "sentence_transformers_version": getattr(
            sentence_transformers, "__version__", "unknown"
        ),
        "torch_version": torch.__version__,
    }
