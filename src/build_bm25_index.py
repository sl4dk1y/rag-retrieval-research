"""Build and persist the Russian Mr. TyDi BM25 index."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

from data_utils import (
    ANALYZER_VERSION,
    BUILD_TIMING_FILENAME,
    CORPUS_PATH,
    DOCIDS_FILENAME,
    INDEX_DIR,
    METADATA_FILENAME,
    STEMMER_LANGUAGE,
    SUCCESS_FILENAME,
    DataFormatError,
    atomic_text_writer,
    check_free_space,
    create_russian_tokenizer,
    index_is_complete,
    iter_jsonl_gz,
    load_bm25_dependencies,
    referenced_relevant_docids,
    require_complete_index,
    require_file,
    require_python_311,
    write_json,
)


BM25_METHOD = "lucene"
BM25_K1 = 1.2
BM25_B = 0.75
def read_corpus(corpus_path: Path) -> tuple[list[str], list[str], float, float]:
    """Stream corpus records into the two arrays required for indexing."""
    texts: list[str] = []
    docids: list[str] = []
    relevant_docids_not_seen = referenced_relevant_docids()
    start = time.perf_counter()

    for record_number, record in enumerate(
        iter_jsonl_gz(corpus_path, description="Чтение корпуса"), start=1
    ):
        try:
            doc_id = record["docid"]
            title = record["title"]
            text = record["text"]
        except KeyError as error:
            raise DataFormatError(
                f"В записи корпуса {record_number:,} отсутствует поле {error.args[0]!r}"
            ) from error
        if not isinstance(doc_id, str) or not doc_id:
            raise DataFormatError(
                f"Некорректный docid в записи корпуса {record_number:,}: {doc_id!r}"
            )
        if not isinstance(title, str) or not isinstance(text, str):
            raise DataFormatError(
                f"title/text должны быть строками в записи корпуса {record_number:,}"
            )
        docids.append(doc_id)
        texts.append(title + " " + text)
        relevant_docids_not_seen.discard(doc_id)

    read_seconds = time.perf_counter() - start
    if not docids:
        raise DataFormatError(f"Корпус не содержит документов: {corpus_path}")
    if relevant_docids_not_seen:
        examples = ", ".join(sorted(relevant_docids_not_seen)[:10])
        raise DataFormatError(
            f"{len(relevant_docids_not_seen):,} релевантных docid из qrels "
            f"отсутствуют в корпусе. Примеры: {examples}"
        )

    print("Проверка уникальности corpus docid...")
    validation_start = time.perf_counter()
    sorted_docids = sorted(docids)
    for previous, current in pairwise(sorted_docids):
        if previous == current:
            raise DataFormatError(f"Повтор docid в корпусе: {current!r}")
    del sorted_docids
    docid_validation_seconds = time.perf_counter() - validation_start
    print(f"Уникальность docid проверена за {docid_validation_seconds:.2f} с")
    return texts, docids, read_seconds, docid_validation_seconds


def _write_docids(path: Path, docids: list[str]) -> None:
    with atomic_text_writer(path) as stream:
        for doc_id in docids:
            stream.write(doc_id)
            stream.write("\n")


def build_index(
    *,
    corpus_path: Path = CORPUS_PATH,
    index_dir: Path = INDEX_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Build an index atomically, or reuse a completed one."""
    require_python_311()
    require_file(corpus_path, "корпус")
    if index_is_complete(index_dir) and not force:
        require_complete_index(index_dir)
        print(f"BM25-индекс уже существует: {index_dir}")
        print("Для перестроения используйте --force.")
        from data_utils import load_json

        return load_json(index_dir / BUILD_TIMING_FILENAME)
    if index_dir.exists() and not force:
        raise RuntimeError(
            f"В {index_dir} найден незавершённый индекс. "
            "Проверьте каталог и запустите перестроение с --force."
        )

    bm25s, Stemmer, Tokenizer = load_bm25_dependencies()
    free_bytes, estimated_bytes = check_free_space(index_dir.parent, corpus_path)
    print(
        "Свободное место: "
        f"{free_bytes / 1024**3:.1f} GiB; "
        f"предварительная оценка: {estimated_bytes / 1024**3:.1f} GiB."
    )

    total_start = time.perf_counter()
    texts, docids, corpus_read_seconds, docid_validation_seconds = read_corpus(
        corpus_path
    )
    print(f"Прочитано документов: {len(docids):,}")

    tokenizer = create_russian_tokenizer(Stemmer, Tokenizer)
    tokenization_start = time.perf_counter()
    text_count = len(texts)

    def consume_texts() -> Any:
        # Release each large source string as soon as it has been tokenized.
        for position, document_text in enumerate(texts):
            texts[position] = ""
            yield document_text

    corpus_tokens = tokenizer.tokenize(
        consume_texts(),
        length=text_count,
        return_as="tuple",
        show_progress=True,
    )
    corpus_tokenization_seconds = time.perf_counter() - tokenization_start
    del texts

    retriever = bm25s.BM25(method=BM25_METHOD, k1=BM25_K1, b=BM25_B)
    index_start = time.perf_counter()
    retriever.index(corpus_tokens, show_progress=True)
    index_build_seconds = time.perf_counter() - index_start
    del corpus_tokens

    temporary_dir = index_dir.with_name(f".{index_dir.name}.building-{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        save_start = time.perf_counter()
        retriever.save(temporary_dir)
        tokenizer.save_vocab(temporary_dir)
        tokenizer.save_stopwords(temporary_dir)
        _write_docids(temporary_dir / DOCIDS_FILENAME, docids)
        index_save_seconds = time.perf_counter() - save_start

        build_timing: dict[str, Any] = {
            "corpus_read_seconds": corpus_read_seconds,
            "corpus_tokenization_seconds": corpus_tokenization_seconds,
            "document_count": len(docids),
            "docid_validation_seconds": docid_validation_seconds,
            "index_build_seconds": index_build_seconds,
            "index_save_seconds": index_save_seconds,
            "index_total_seconds": time.perf_counter() - total_start,
        }
        metadata: dict[str, Any] = {
            "analyzer_version": ANALYZER_VERSION,
            "bm25": {"b": BM25_B, "k1": BM25_K1, "method": BM25_METHOD},
            "corpus": str(corpus_path.resolve()),
            "corpus_size_bytes": corpus_path.stat().st_size,
            "document_count": len(docids),
            "format_version": 1,
            "packages": {
                "PyStemmer": importlib.metadata.version("PyStemmer"),
                "bm25s": importlib.metadata.version("bm25s"),
            },
            "python": sys.version.split()[0],
            "stemmer": {
                "implementation": "PyStemmer/Snowball",
                "language": STEMMER_LANGUAGE,
            },
            "text_fields": "title + ' ' + text",
            "tokenizer": {"stopwords": []},
        }
        write_json(temporary_dir / BUILD_TIMING_FILENAME, build_timing)
        write_json(temporary_dir / METADATA_FILENAME, metadata)
        with atomic_text_writer(temporary_dir / SUCCESS_FILENAME) as stream:
            stream.write("ok\n")

        if index_dir.exists():
            if not force:
                raise FileExistsError(
                    f"Каталог индекса появился во время построения: {index_dir}"
                )
            shutil.rmtree(index_dir)
        os.replace(temporary_dir, index_dir)
    except OSError as error:
        if error.errno == 28:
            raise OSError(
                f"Недостаточно свободного места для сохранения индекса в {index_dir}"
            ) from error
        raise
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    print(f"Индекс сохранён: {index_dir}")
    print(f"Чтение корпуса:     {corpus_read_seconds:.2f} с")
    print(f"Токенизация:        {corpus_tokenization_seconds:.2f} с")
    print(f"Построение индекса: {index_build_seconds:.2f} с")
    return build_timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="перестроить уже существующий индекс"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except (DataFormatError, FileNotFoundError, OSError, RuntimeError) as error:
        print(f"ОШИБКА: {error}", file=sys.stderr)
        raise SystemExit(1) from error
