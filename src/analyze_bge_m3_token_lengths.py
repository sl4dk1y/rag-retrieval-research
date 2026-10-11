"""Analyze BGE-M3 token-length distribution on the raw Mr. TyDi Russian corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian" / "corpus.jsonl.gz"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_NAME = "BAAI/bge-m3"
FULL_CORPUS_DOCUMENTS = 9_597_504
THRESHOLDS = (128, 256, 512, 1024, 8192)
PERCENTILES = (50, 90, 95, 99, 99.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def iter_texts(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Не найден корпус: {path}")

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                title = record["title"]
                text = record["text"]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Некорректная запись корпуса #{number}"
                ) from error

            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError(
                    f"title/text должны быть строками в записи #{number}"
                )

            yield f"{title} {text}"


def batched(iterator, batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    args = parse_args()

    if args.max_docs is not None and args.max_docs <= 0:
        raise ValueError("--max-docs должен быть > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть > 0")

    target_docs = args.max_docs or FULL_CORPUS_DOCUMENTS

    print(f"Model tokenizer: {args.model}")
    print(f"Documents:       {target_docs:,}")
    print(f"Batch size:      {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    lengths: list[int] = []
    processed = 0
    started = time.perf_counter()

    progress = tqdm(total=target_docs, unit="doc", desc="Tokenizing")

    for texts in batched(iter_texts(CORPUS_PATH), args.batch_size):
        remaining = target_docs - processed
        if remaining <= 0:
            break
        if len(texts) > remaining:
            texts = texts[:remaining]

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        lengths.extend(batch_lengths)

        processed += len(batch_lengths)
        progress.update(len(batch_lengths))

        if processed >= target_docs:
            break

    progress.close()

    if processed != target_docs:
        raise RuntimeError(
            f"Получено {processed:,} документов вместо {target_docs:,}"
        )

    elapsed = time.perf_counter() - started
    values = np.asarray(lengths, dtype=np.int32)

    percentiles = {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in PERCENTILES
    }

    thresholds = {
        str(threshold): {
            "documents_over": int(np.sum(values > threshold)),
            "fraction_over": float(np.mean(values > threshold)),
        }
        for threshold in THRESHOLDS
    }

    result = {
        "model": args.model,
        "text_format": 'title + " " + text',
        "add_special_tokens": True,
        "processed_docs": processed,
        "max_docs": args.max_docs,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "documents_per_second": processed / elapsed,
        "minimum_tokens": int(values.min()),
        "maximum_tokens": int(values.max()),
        "mean_tokens": float(values.mean()),
        "percentiles": percentiles,
        "thresholds": thresholds,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = str(processed) if args.max_docs is not None else "full"
    output_path = RESULTS_DIR / f"bge_m3_token_lengths_{suffix}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== BGE-M3 token lengths ===")
    print(f"Documents:       {processed:,}")
    print(f"Elapsed:         {elapsed:.2f} s")
    print(f"Documents/sec:   {processed / elapsed:.2f}")
    print(f"Mean tokens:     {values.mean():.2f}")
    print(f"Median tokens:   {percentiles['50']:.0f}")
    print(f"P90:             {percentiles['90']:.0f}")
    print(f"P95:             {percentiles['95']:.0f}")
    print(f"P99:             {percentiles['99']:.0f}")
    print(f"P99.9:           {percentiles['99.9']:.0f}")
    print(f"Max tokens:      {values.max():,}")

    for threshold in THRESHOLDS:
        stats = thresholds[str(threshold)]
        print(
            f"> {threshold:4d}:         "
            f"{stats['documents_over']:,} "
            f"({stats['fraction_over'] * 100:.4f}%)"
        )

    print(f"Result saved:    {output_path}")


if __name__ == "__main__":
    main()
