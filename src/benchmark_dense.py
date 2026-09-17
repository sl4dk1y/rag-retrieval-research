"""Benchmark dense encoding models on Russian Mr. TyDi.

Diagnostic only: embeddings are not persisted and no FAISS index is built.
"""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian" / "corpus.jsonl.gz"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_NAME = "BAAI/bge-m3"
FULL_CORPUS_DOCUMENTS = 9_597_504


@dataclass
class BenchmarkResult:
    model: str
    device: str
    max_docs: int
    processed_docs: int
    batch_size: int
    max_seq_length: int
    embedding_dimension: int
    model_load_seconds: float
    encoding_seconds: float
    corpus_loop_seconds: float
    total_seconds: float
    encoding_documents_per_second: float
    end_to_end_documents_per_second: float
    estimated_full_corpus_seconds: float
    estimated_full_corpus_hours: float
    peak_rss_bytes: int
    peak_rss_gib: float
    torch_version: str
    mps_available: bool


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS/Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def choose_device(requested: str) -> str:
    """Resolve auto/mps/cpu into an available device."""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Запрошен MPS, но torch.backends.mps.is_available() вернул False."
        )
    return requested


def synchronize_device(device: str) -> None:
    """Synchronize asynchronous accelerator work before timing boundaries."""
    if device == "mps":
        torch.mps.synchronize()


def iter_corpus_texts(path: Path, max_docs: int) -> Iterator[str]:
    """Yield exactly `title + space + text` from the gzip JSONL corpus."""
    if not path.is_file():
        raise FileNotFoundError(f"Не найден корпус: {path}")

    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            emitted = 0
            for number, line in enumerate(stream, start=1):
                if emitted >= max_docs:
                    break
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Некорректный JSON в записи корпуса #{number}"
                    ) from error

                try:
                    title = record["title"]
                    text = record["text"]
                except KeyError as error:
                    raise ValueError(
                        f"В записи корпуса #{number} отсутствует поле {error.args[0]!r}"
                    ) from error

                if not isinstance(title, str) or not isinstance(text, str):
                    raise ValueError(
                        f"title/text должны быть строками в записи корпуса #{number}"
                    )

                emitted += 1
                yield title + " " + text
    except (OSError, EOFError) as error:
        raise RuntimeError(f"Не удалось прочитать gzip-корпус: {path}") from error


def batched(iterator: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    """Collect an iterator into small batches without loading the full corpus."""
    batch: list[str] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark dense encoding on a subset of Russian Mr. TyDi "
            "without saving embeddings."
        )
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10_000,
        help="число первых документов для benchmark (по умолчанию: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size для model.encode (по умолчанию: 16)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="устройство вычислений; auto предпочитает Apple MPS",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Hugging Face model id (по умолчанию: {MODEL_NAME})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "ограничение длины входа модели; "
            "по умолчанию используется model.max_seq_length"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_docs <= 0:
        raise ValueError("--max-docs должен быть положительным целым числом")
    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть положительным целым числом")
    if args.max_seq_length is not None and args.max_seq_length <= 0:
        raise ValueError("--max-seq-length должен быть > 0")

    device = choose_device(args.device)
    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(f"Max docs:   {args.max_docs:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"MPS:        {torch.backends.mps.is_available()}")
    print()

    total_start = time.perf_counter()

    print("Загрузка модели...")
    load_start = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    model_load_seconds = time.perf_counter() - load_start
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length

    embedding_dimension = model.get_embedding_dimension()
    if embedding_dimension is None:
        raise RuntimeError("Не удалось определить размерность embeddings модели")

    print(
        f"Модель загружена за {model_load_seconds:.2f} с; "
        f"dimension={embedding_dimension}"
    )
    print(f"Max seq length: {model.max_seq_length}")

    print("Warm-up...")
    model.encode(
        ["Тестовый документ для прогрева модели."],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    synchronize_device(device)

    encoding_seconds = 0.0
    processed_docs = 0
    texts = iter_corpus_texts(CORPUS_PATH, args.max_docs)
    progress = tqdm(total=args.max_docs, unit="doc", desc="Dense encoding")
    corpus_loop_start = time.perf_counter()

    for batch in batched(texts, args.batch_size):
        synchronize_device(device)
        encode_start = time.perf_counter()
        embeddings = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        synchronize_device(device)
        encoding_seconds += time.perf_counter() - encode_start

        if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dimension:
            raise RuntimeError(
                f"Неожиданная форма embeddings: {embeddings.shape}; "
                f"ожидалось (*, {embedding_dimension})"
            )
        processed_docs += len(batch)
        progress.update(len(batch))
        del embeddings

    synchronize_device(device)
    progress.close()
    corpus_loop_seconds = time.perf_counter() - corpus_loop_start

    if processed_docs == 0:
        raise RuntimeError("Из корпуса не было прочитано ни одного документа")

    if processed_docs < args.max_docs:
        print(
            f"Предупреждение: корпус закончился после {processed_docs:,} документов."
        )

    total_seconds = time.perf_counter() - total_start
    encoding_documents_per_second = processed_docs / encoding_seconds
    end_to_end_documents_per_second = processed_docs / corpus_loop_seconds
    estimated_seconds = FULL_CORPUS_DOCUMENTS / end_to_end_documents_per_second
    rss = peak_rss_bytes()

    result = BenchmarkResult(
        model=args.model,
        device=device,
        max_docs=args.max_docs,
        processed_docs=processed_docs,
        batch_size=args.batch_size,
        max_seq_length=int(model.max_seq_length),
        embedding_dimension=embedding_dimension,
        model_load_seconds=model_load_seconds,
        encoding_seconds=encoding_seconds,
        corpus_loop_seconds=corpus_loop_seconds,
        total_seconds=total_seconds,
        encoding_documents_per_second=encoding_documents_per_second,
        end_to_end_documents_per_second=end_to_end_documents_per_second,
        estimated_full_corpus_seconds=estimated_seconds,
        estimated_full_corpus_hours=estimated_seconds / 3600,
        peak_rss_bytes=rss,
        peak_rss_gib=rss / 1024**3,
        torch_version=torch.__version__,
        mps_available=torch.backends.mps.is_available(),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model_name = args.model.replace("/", "__")
    output_path = RESULTS_DIR / (
        f"dense_benchmark_{safe_model_name}_{processed_docs}"
        f"_batch{args.batch_size}_seq{model.max_seq_length}.json"
    )
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== Dense benchmark ===")
    print(f"Documents:             {processed_docs:,}")
    print(f"Embedding dimension:   {embedding_dimension}")
    print(f"Model load:            {model_load_seconds:.2f} s")
    print(f"Encoding time:         {encoding_seconds:.2f} s")
    print(f"Corpus loop time:      {corpus_loop_seconds:.2f} s")
    print(f"Encoding docs/sec:     {encoding_documents_per_second:.2f}")
    print(f"End-to-end docs/sec:   {end_to_end_documents_per_second:.2f}")
    print(f"Peak RAM (RSS):        {rss / 1024**3:.2f} GiB")
    print(
        "Estimated full corpus: "
        f"{estimated_seconds / 3600:.2f} h ({FULL_CORPUS_DOCUMENTS:,} docs)"
    )
    print(f"Result saved:          {output_path}")


if __name__ == "__main__":
    main()
