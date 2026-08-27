"""Build reproducible final result tables from stored experiment artifacts.

The script aggregates already completed BM25, Dense Retrieval, Hybrid Retrieval,
timing, ablation, and error-analysis JSON files. It does not run retrieval and
does not tune any parameters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final"

FINAL_JSON = OUTPUT_DIR / "final_results.json"
TEST_COMPARISON_CSV = OUTPUT_DIR / "test_comparison.csv"
DEV_COMPARISON_CSV = OUTPUT_DIR / "dev_comparison.csv"
DENSE_NPROBE_CSV = OUTPUT_DIR / "dense_nprobe_ablation.csv"
HYBRID_WEIGHT_CSV = OUTPUT_DIR / "hybrid_weight_ablation.csv"
LATENCY_CSV = OUTPUT_DIR / "latency_summary.csv"
ERROR_ANALYSIS_CSV = OUTPUT_DIR / "error_analysis_summary.csv"


METRIC_ORDER = ("Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_metrics(path: Path) -> dict[str, float]:
    payload = load_json(path)
    metrics = payload.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics object in {path}")

    result: dict[str, float] = {}
    for name in METRIC_ORDER:
        if name not in metrics:
            raise ValueError(f"Metric {name!r} is missing in {path}")
        result[name] = float(metrics[name])

    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(method: str, metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method}
    row.update(extra)
    for metric in METRIC_ORDER:
        row[metric] = metrics[metric]
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Final frozen test comparison
    # ------------------------------------------------------------------
    bm25_test = load_metrics(TABLES_DIR / "bm25_test_metrics.json")
    dense_test = load_metrics(
        TABLES_DIR / "dense_e5_small_test_nprobe256_metrics.json"
    )
    hybrid_test = load_metrics(
        TABLES_DIR / "hybrid_rrf_w0025_10_test_metrics.json"
    )

    test_rows = [
        metric_row("BM25", bm25_test),
        metric_row("Dense E5 (nprobe=256)", dense_test),
        metric_row("Hybrid RRF (BM25 weight=0.025)", hybrid_test),
    ]

    write_csv(
        TEST_COMPARISON_CSV,
        test_rows,
        ["method", *METRIC_ORDER],
    )

    # ------------------------------------------------------------------
    # Final dev comparison
    # ------------------------------------------------------------------
    bm25_dev = load_metrics(TABLES_DIR / "bm25_dev_metrics.json")
    dense_dev_256 = load_metrics(
        TABLES_DIR / "dense_e5_small_dev_nprobe256_metrics.json"
    )
    hybrid_dev_0025 = load_metrics(
        TABLES_DIR / "hybrid_rrf_w0025_10_dev_metrics.json"
    )

    dev_rows = [
        metric_row("BM25", bm25_dev),
        metric_row("Dense E5 (nprobe=256)", dense_dev_256),
        metric_row("Hybrid RRF (BM25 weight=0.025)", hybrid_dev_0025),
    ]

    write_csv(
        DEV_COMPARISON_CSV,
        dev_rows,
        ["method", *METRIC_ORDER],
    )

    # ------------------------------------------------------------------
    # Dense nprobe ablation
    # ------------------------------------------------------------------
    dense_nprobe_files = [
        (64, TABLES_DIR / "dense_e5_small_dev_nprobe64_metrics.json"),
        (128, TABLES_DIR / "dense_e5_small_dev_nprobe128_metrics.json"),
        (256, TABLES_DIR / "dense_e5_small_dev_nprobe256_metrics.json"),
        (512, TABLES_DIR / "dense_e5_small_dev_nprobe512_metrics.json"),
    ]

    dense_nprobe_rows: list[dict[str, Any]] = []
    for nprobe, path in dense_nprobe_files:
        metrics = load_metrics(path)
        row: dict[str, Any] = {"nprobe": nprobe}
        row.update(metrics)
        dense_nprobe_rows.append(row)

    write_csv(
        DENSE_NPROBE_CSV,
        dense_nprobe_rows,
        ["nprobe", *METRIC_ORDER],
    )

    # ------------------------------------------------------------------
    # Hybrid weight ablation
    # ------------------------------------------------------------------
    hybrid_weight_files = [
        (1.0, TABLES_DIR / "hybrid_rrf_equal_dev_metrics.json"),
        (0.5, TABLES_DIR / "hybrid_rrf_w05_10_dev_metrics.json"),
        (0.25, TABLES_DIR / "hybrid_rrf_w025_10_dev_metrics.json"),
        (0.1, TABLES_DIR / "hybrid_rrf_w01_10_dev_metrics.json"),
        (0.05, TABLES_DIR / "hybrid_rrf_w005_10_dev_metrics.json"),
        (0.025, TABLES_DIR / "hybrid_rrf_w0025_10_dev_metrics.json"),
        (0.01, TABLES_DIR / "hybrid_rrf_w001_10_dev_metrics.json"),
    ]

    hybrid_weight_rows: list[dict[str, Any]] = []
    for bm25_weight, path in hybrid_weight_files:
        metrics = load_metrics(path)
        row: dict[str, Any] = {
            "bm25_weight": bm25_weight,
            "dense_weight": 1.0,
            "rrf_k": 60,
        }
        row.update(metrics)
        hybrid_weight_rows.append(row)

    write_csv(
        HYBRID_WEIGHT_CSV,
        hybrid_weight_rows,
        ["bm25_weight", "dense_weight", "rrf_k", *METRIC_ORDER],
    )

    # ------------------------------------------------------------------
    # Latency / timing summary
    # ------------------------------------------------------------------
    bm25_test_timing = load_json(TABLES_DIR / "bm25_test_timing.json")
    dense_test_timing = load_json(
        TABLES_DIR / "dense_e5_small_test_nprobe256_timing.json"
    )

    bm25_search_seconds = float(
        bm25_test_timing["query_retrieval_seconds"]
    )
    bm25_query_count = int(
        bm25_test_timing["query_count"]
    )
    bm25_avg_seconds = float(
        bm25_test_timing["average_query_seconds"]
    )

    dense_search_seconds = float(dense_test_timing["search_seconds"])
    dense_query_count = int(dense_test_timing["query_count"])
    dense_avg_seconds = float(dense_test_timing["average_search_seconds_per_query"])
    dense_encode_seconds = float(dense_test_timing["query_encode_seconds"])

    latency_rows = [
        {
            "method": "BM25",
            "split": "test",
            "query_count": bm25_query_count,
            "query_encoding_seconds": "",
            "search_seconds": bm25_search_seconds,
            "average_search_ms_per_query": bm25_avg_seconds * 1000.0,
        },
        {
            "method": "Dense E5 (nprobe=256)",
            "split": "test",
            "query_count": dense_query_count,
            "query_encoding_seconds": dense_encode_seconds,
            "search_seconds": dense_search_seconds,
            "average_search_ms_per_query": dense_avg_seconds * 1000.0,
        },
    ]

    write_csv(
        LATENCY_CSV,
        latency_rows,
        [
            "method",
            "split",
            "query_count",
            "query_encoding_seconds",
            "search_seconds",
            "average_search_ms_per_query",
        ],
    )

    # ------------------------------------------------------------------
    # Error-analysis summary
    # ------------------------------------------------------------------
    error_payload = load_json(ANALYSIS_DIR / "error_analysis_test_summary.json")
    categories = error_payload.get("category_counts", {})
    dense_vs_hybrid = error_payload.get("dense_vs_hybrid", {})

    if not isinstance(categories, dict) or not isinstance(dense_vs_hybrid, dict):
        raise ValueError("Invalid error-analysis summary structure")

    error_rows: list[dict[str, Any]] = []

    for name, count in sorted(categories.items()):
        error_rows.append(
            {
                "section": "category_counts",
                "name": name,
                "count": int(count),
                "share": int(count) / int(error_payload["query_count"]),
            }
        )

    for name, count in sorted(dense_vs_hybrid.items()):
        error_rows.append(
            {
                "section": "dense_vs_hybrid",
                "name": name,
                "count": int(count),
                "share": int(count) / int(error_payload["query_count"]),
            }
        )

    write_csv(
        ERROR_ANALYSIS_CSV,
        error_rows,
        ["section", "name", "count", "share"],
    )

    # ------------------------------------------------------------------
    # Consolidated machine-readable JSON
    # ------------------------------------------------------------------
    final_payload = {
        "protocol": {
            "dataset": "Mr. TyDi Russian",
            "corpus_documents": 9_597_504,
            "dev_used_for_parameter_selection": True,
            "test_used_for_parameter_tuning": False,
            "final_test_queries": 995,
        },
        "frozen_configurations": {
            "bm25": {
                "variant": "Lucene BM25",
                "k1": 1.2,
                "b": 0.75,
                "top_k": 100,
            },
            "dense": {
                "model": "intfloat/multilingual-e5-small",
                "embedding_dimension": 384,
                "faiss": "IVFScalarQuantizer(SQ8, inner-product)",
                "nlist": 2048,
                "nprobe": 256,
                "top_k": 100,
            },
            "hybrid": {
                "method": "weighted Reciprocal Rank Fusion",
                "rrf_k": 60,
                "bm25_weight": 0.025,
                "dense_weight": 1.0,
                "dense_nprobe": 256,
                "top_k": 100,
            },
        },
        "dev_comparison": {
            row["method"]: {metric: row[metric] for metric in METRIC_ORDER}
            for row in dev_rows
        },
        "test_comparison": {
            row["method"]: {metric: row[metric] for metric in METRIC_ORDER}
            for row in test_rows
        },
        "dense_nprobe_ablation": dense_nprobe_rows,
        "hybrid_weight_ablation": hybrid_weight_rows,
        "latency": latency_rows,
        "error_analysis": {
            "query_count": int(error_payload["query_count"]),
            "cutoff": int(error_payload["cutoff"]),
            "category_counts": categories,
            "dense_vs_hybrid": dense_vs_hybrid,
        },
        "artifacts": {
            "test_comparison_csv": "results/final/test_comparison.csv",
            "dev_comparison_csv": "results/final/dev_comparison.csv",
            "dense_nprobe_ablation_csv": "results/final/dense_nprobe_ablation.csv",
            "hybrid_weight_ablation_csv": "results/final/hybrid_weight_ablation.csv",
            "latency_summary_csv": "results/final/latency_summary.csv",
            "error_analysis_summary_csv": "results/final/error_analysis_summary.csv",
        },
    }

    FINAL_JSON.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Final results aggregation ===")
    print(f"Final JSON:              {FINAL_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Test comparison CSV:     {TEST_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Dev comparison CSV:      {DEV_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Dense nprobe CSV:        {DENSE_NPROBE_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Hybrid weight CSV:       {HYBRID_WEIGHT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Latency CSV:             {LATENCY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Error analysis CSV:      {ERROR_ANALYSIS_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
