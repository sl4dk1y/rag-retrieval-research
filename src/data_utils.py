"""Shared data and filesystem helpers for the Mr. TyDi BM25 baseline."""

from __future__ import annotations

import errno
import gzip
import json
import os
import shutil
import sys
import tempfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian"
IR_DATA_DIR = DATASET_DIR / "ir-format-data"
CORPUS_PATH = DATASET_DIR / "corpus.jsonl.gz"
INDEX_DIR = PROJECT_ROOT / "indexes" / "bm25"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

SUPPORTED_SPLITS = ("train", "dev", "test")
DOCIDS_FILENAME = "docids.txt"
METADATA_FILENAME = "metadata.json"
BUILD_TIMING_FILENAME = "build_timing.json"
SUCCESS_FILENAME = "_SUCCESS"
STEMMER_LANGUAGE = "russian"
ANALYZER_VERSION = 1


class DataFormatError(ValueError):
    """Raised when a dataset, mapping, or run file has an invalid format."""


class CorruptGzipError(DataFormatError):
    """Raised when a gzip file cannot be decompressed completely."""


def load_bm25_dependencies() -> tuple[Any, Any, Any]:
    """Load optional BM25 dependencies with one actionable error message."""
    try:
        import bm25s
        import Stemmer
        from bm25s.tokenization import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "Не установлены зависимости BM25. Выполните: "
            "pip install -r requirements.txt"
        ) from error
    return bm25s, Stemmer, Tokenizer


def create_russian_tokenizer(stemmer_module: Any, tokenizer_class: Any) -> Any:
    """Create the shared analyzer used for both documents and queries."""
    return tokenizer_class(
        stemmer=stemmer_module.Stemmer(STEMMER_LANGUAGE), stopwords=[]
    )


def require_python_311() -> None:
    """Fail early when an unsupported Python interpreter is used."""
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "Требуется Python 3.11 или новее; сейчас используется "
            f"{sys.version.split()[0]}."
        )


def require_file(path: Path, description: str = "файл") -> Path:
    """Return *path* if it is a non-empty regular file, otherwise fail clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден {description}: {path}")
    if path.stat().st_size == 0:
        raise DataFormatError(f"Пустой {description}: {path}")
    return path


def split_paths(split: str) -> tuple[Path, Path]:
    """Return topics and qrels paths for a supported split."""
    if split not in SUPPORTED_SPLITS:
        allowed = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Неизвестный split {split!r}; допустимы: {allowed}")
    return (
        IR_DATA_DIR / f"topics.{split}.txt",
        IR_DATA_DIR / f"qrels.{split}.txt",
    )


def validate_split_files(split: str) -> tuple[Path, Path]:
    """Validate and return the topics and qrels files for *split*."""
    topics_path, qrels_path = split_paths(split)
    require_file(topics_path, f"topics для split={split}")
    require_file(qrels_path, f"qrels для split={split}")
    return topics_path, qrels_path


def iter_jsonl_gz(path: Path, *, description: str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from gzip JSONL without retaining records in memory."""
    require_file(path, description)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            progress = tqdm(stream, desc=description, unit=" док.", dynamic_ncols=True)
            for line_number, line in enumerate(progress, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataFormatError(
                        f"Некорректный JSON в {path}, строка {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise DataFormatError(
                        f"Ожидался JSON-объект в {path}, строка {line_number}"
                    )
                yield record
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise CorruptGzipError(f"Повреждён gzip-файл {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise DataFormatError(f"Файл {path} не является корректным UTF-8: {error}") from error


def load_topics(path: Path) -> list[tuple[str, str]]:
    """Load a Mr. TyDi topics TSV while preserving its query order."""
    require_file(path, "topics-файл")
    topics: list[tuple[str, str]] = []
    seen_query_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1].strip():
                raise DataFormatError(
                    f"Некорректная строка topics в {path}:{line_number}"
                )
            query_id, query = parts[0], parts[1]
            if query_id in seen_query_ids:
                raise DataFormatError(f"Повтор query_id={query_id!r} в {path}")
            seen_query_ids.add(query_id)
            topics.append((query_id, query))
    if not topics:
        raise DataFormatError(f"Topics-файл не содержит запросов: {path}")
    return topics


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load TREC qrels as query_id -> doc_id -> integer relevance."""
    require_file(path, "qrels-файл")
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise DataFormatError(
                    f"Ожидалось 4 поля qrels в {path}:{line_number}, получено {len(parts)}"
                )
            query_id, _iteration, doc_id, relevance_text = parts
            try:
                relevance = int(relevance_text)
            except ValueError as error:
                raise DataFormatError(
                    f"Некорректная релевантность в {path}:{line_number}: "
                    f"{relevance_text!r}"
                ) from error
            query_qrels = qrels.setdefault(query_id, {})
            if doc_id in query_qrels:
                raise DataFormatError(
                    f"Повтор qrels для query_id={query_id!r}, doc_id={doc_id!r}"
                )
            query_qrels[doc_id] = relevance
    if not qrels:
        raise DataFormatError(f"Qrels-файл не содержит оценок: {path}")
    return qrels


def referenced_relevant_docids() -> set[str]:
    """Collect all positively relevant docids used by available Mr. TyDi qrels."""
    docids: set[str] = set()
    for split in SUPPORTED_SPLITS:
        _topics_path, qrels_path = validate_split_files(split)
        qrels = load_qrels(qrels_path)
        docids.update(
            doc_id
            for query_qrels in qrels.values()
            for doc_id, relevance in query_qrels.items()
            if relevance > 0
        )
    return docids


def read_docids(path: Path, *, expected_count: int | None = None) -> list[str]:
    """Load the internal-index-to-Mr.-TyDi-docid mapping."""
    require_file(path, "mapping docids")
    docids: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in tqdm(stream, desc="Загрузка mapping docid", unit=" док."):
            doc_id = line.rstrip("\n\r")
            if not doc_id:
                raise DataFormatError(f"Пустой docid в mapping: {path}")
            docids.append(doc_id)
    if expected_count is not None and len(docids) != expected_count:
        raise DataFormatError(
            "Число docid не совпадает с размером индекса: "
            f"mapping={len(docids):,}, index={expected_count:,}"
        )
    return docids


def validate_referenced_docids(mapping_path: Path, referenced: set[str]) -> None:
    """Verify that all referenced docids occur in a mapping, using bounded memory."""
    if not referenced:
        return
    missing = set(referenced)
    require_file(mapping_path, "mapping docids")
    with mapping_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            missing.discard(line.rstrip("\n\r"))
            if not missing:
                return
    examples = ", ".join(sorted(missing)[:10])
    suffix = "" if len(missing) <= 10 else f" … и ещё {len(missing) - 10}"
    raise DataFormatError(
        f"{len(missing):,} docid отсутствуют в mapping индекса: {examples}{suffix}"
    )


def load_json(path: Path, *, description: str = "JSON-файл") -> dict[str, Any]:
    """Read a JSON object with consistent validation errors."""
    require_file(path, description)
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise DataFormatError(f"Некорректный JSON в {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataFormatError(f"Ожидался JSON-объект в {path}")
    return value


def _raise_friendly_os_error(error: OSError, path: Path) -> None:
    if error.errno == errno.ENOSPC:
        raise OSError(f"Недостаточно свободного места для записи {path}") from error
    raise error


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Write a text file atomically and report ENOSPC clearly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        _raise_friendly_os_error(error, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write readable, deterministic UTF-8 JSON."""
    with atomic_text_writer(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def check_free_space(target_dir: Path, source_path: Path) -> tuple[int, int]:
    """Fail if a conservative index-size estimate exceeds available disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    # A sparse BM25 index plus temporary arrays can exceed the compressed corpus.
    required_bytes = max(2 * 1024**3, source_path.stat().st_size * 4)
    if free_bytes < required_bytes:
        raise OSError(
            "Недостаточно свободного места для безопасного построения индекса: "
            f"доступно {free_bytes / 1024**3:.1f} GiB, "
            f"оценка минимума {required_bytes / 1024**3:.1f} GiB."
        )
    return free_bytes, required_bytes


def index_is_complete(index_dir: Path = INDEX_DIR) -> bool:
    """Return whether an index has all baseline-owned completion markers."""
    required = (
        index_dir / SUCCESS_FILENAME,
        index_dir / METADATA_FILENAME,
        index_dir / BUILD_TIMING_FILENAME,
        index_dir / DOCIDS_FILENAME,
        index_dir / "data.csc.index.npy",
        index_dir / "indices.csc.index.npy",
        index_dir / "indptr.csc.index.npy",
        index_dir / "params.index.json",
        index_dir / "vocab.index.json",
        index_dir / "stopwords.tokenizer.json",
        index_dir / "vocab.tokenizer.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def require_complete_index(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    """Validate index completion markers and return its metadata."""
    if not index_is_complete(index_dir):
        raise FileNotFoundError(
            f"Готовый BM25-индекс не найден в {index_dir}. "
            "Сначала запустите: python src/build_bm25_index.py"
        )
    metadata = load_json(index_dir / METADATA_FILENAME, description="metadata индекса")
    document_count = metadata.get("document_count")
    if not isinstance(document_count, int) or document_count <= 0:
        raise DataFormatError(
            f"Некорректный document_count в {index_dir / METADATA_FILENAME}"
        )
    stemmer = metadata.get("stemmer")
    tokenizer = metadata.get("tokenizer")
    if (
        metadata.get("analyzer_version") != ANALYZER_VERSION
        or not isinstance(stemmer, dict)
        or stemmer.get("language") != STEMMER_LANGUAGE
        or not isinstance(tokenizer, dict)
        or tokenizer.get("stopwords") != []
    ):
        raise DataFormatError(
            "Конфигурация анализатора в сохранённом индексе не совпадает с кодом. "
            "Перестройте индекс: python src/build_bm25_index.py --force"
        )
    return metadata
