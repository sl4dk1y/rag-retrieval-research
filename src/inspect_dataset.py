"""Первичная проверка локальной копии русскоязычного Mr. TyDi."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mr-tydi-russian"
IR_DATA_DIR = DATASET_DIR / "ir-format-data"


def read_first_json_record(path: Path) -> dict[str, Any]:
    """Прочитать первую непустую запись из файла JSONL.GZ."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return json.loads(line)

    raise ValueError(f"Файл не содержит записей: {path}")


def count_gzip_lines(path: Path) -> int:
    """Подсчитать количество непустых строк в файле GZIP."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def count_text_lines(path: Path) -> int:
    """Подсчитать количество непустых строк в текстовом файле."""
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def find_ir_file(prefix: str, split: str) -> Path:
    """Найти topics или qrels для указанной выборки."""
    candidates = sorted(IR_DATA_DIR.glob(f"{prefix}.{split}.*"))

    if not candidates:
        raise FileNotFoundError(
            f"Не найден файл {prefix}.{split}.* в каталоге {IR_DATA_DIR}"
        )

    return candidates[0]


def validate_dataset() -> None:
    """Проверить наличие обязательных файлов датасета."""
    required = [
        DATASET_DIR / "corpus.jsonl.gz",
        DATASET_DIR / "train.jsonl.gz",
        DATASET_DIR / "dev.jsonl.gz",
        DATASET_DIR / "test.jsonl.gz",
    ]

    missing = [path for path in required if not path.is_file()]

    if missing:
        files = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Не найдены обязательные файлы:\n{files}")

    if not IR_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Не найден каталог: {IR_DATA_DIR}")


def print_split_statistics(split: str) -> None:
    """Вывести статистику для train, dev или test."""
    jsonl_path = DATASET_DIR / f"{split}.jsonl.gz"
    topics_path = find_ir_file("topics", split)
    qrels_path = find_ir_file("qrels", split)

    print(f"{split.upper()}:")
    print(f"  JSONL-записей: {count_gzip_lines(jsonl_path):,}")
    print(f"  Запросов:       {count_text_lines(topics_path):,}")
    print(f"  Qrels:          {count_text_lines(qrels_path):,}")
    print(f"  Topics-файл:    {topics_path.name}")
    print(f"  Qrels-файл:     {qrels_path.name}")
    print()


def main() -> None:
    """Проверить датасет и вывести примеры записей."""
    validate_dataset()

    print(f"Каталог датасета:\n{DATASET_DIR}\n")

    for split in ("train", "dev", "test"):
        print_split_statistics(split)

    print("Первая запись корпуса:")
    document = read_first_json_record(DATASET_DIR / "corpus.jsonl.gz")
    print(json.dumps(document, ensure_ascii=False, indent=2)[:2500])

    print("\nПервая запись DEV:")
    dev_record = read_first_json_record(DATASET_DIR / "dev.jsonl.gz")
    print(json.dumps(dev_record, ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
