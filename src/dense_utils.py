"""Shared utilities for multilingual-E5 Dense Retrieval."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian"
CORPUS_PATH = DATASET_DIR / "corpus.jsonl.gz"
IR_DATA_DIR = DATASET_DIR / "ir-format-data"

DENSE_INDEX_DIR = PROJECT_ROOT / "indexes" / "dense-e5-small"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 8
DEFAULT_TOP_K = 100
FULL_CORPUS_DOCUMENTS = 9_597_504


def passage_text(title: str, text: str) -> str:
    """Apply the passage prefix required by E5 retrieval models."""
    return f"passage: {title} {text}"


def query_text(text: str) -> str:
    """Apply the query prefix required by E5 retrieval models."""
    return f"query: {text}"


def iter_corpus(path: Path = CORPUS_PATH) -> Iterator[tuple[str, str]]:
    """Yield (docid, prefixed passage text) from corpus.jsonl.gz."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден корпус: {path}")

    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Некорректный JSON в записи корпуса #{number}"
                    ) from error

                try:
                    docid = record["docid"]
                    title = record["title"]
                    text = record["text"]
                except KeyError as error:
                    raise ValueError(
                        f"В записи корпуса #{number} отсутствует поле {error.args[0]!r}"
                    ) from error

                if not all(isinstance(value, str) for value in (docid, title, text)):
                    raise ValueError(
                        f"docid/title/text должны быть строками в записи #{number}"
                    )
                if not docid:
                    raise ValueError(f"Пустой docid в записи корпуса #{number}")

                yield docid, passage_text(title, text)
    except (OSError, EOFError) as error:
        raise RuntimeError(f"Не удалось прочитать gzip-корпус: {path}") from error


def batched_corpus(
    iterator: Iterator[tuple[str, str]], batch_size: int
) -> Iterator[tuple[list[str], list[str]]]:
    """Batch corpus entries without loading the full corpus into RAM."""
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


def find_topics_file(split: str) -> Path:
    candidates = sorted(IR_DATA_DIR.glob(f"topics.{split}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"Не найден topics для split={split!r} в {IR_DATA_DIR}"
        )
    return candidates[0]


def load_topics(split: str) -> list[tuple[str, str]]:
    """Load and E5-prefix query texts from a Mr. TyDi topics file."""
    path = find_topics_file(split)
    topics: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Некорректная строка topics #{number} в {path}: {line!r}"
                )
            query_id, text = parts
            topics.append((query_id, query_text(text)))

    if not topics:
        raise ValueError(f"Файл topics пуст: {path}")
    return topics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_metadata(index_dir: Path = DENSE_INDEX_DIR) -> dict:
    path = index_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Не найден metadata dense-индекса: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
