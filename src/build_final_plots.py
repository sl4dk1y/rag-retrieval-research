"""Build publication-ready figures from aggregated final result CSV files.

The script reads files produced by `src/build_final_results.py` and creates
separate figures in `results/final/figures/`. It does not recompute retrieval
metrics or tune parameters.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "results" / "final"
FIGURES_DIR = FINAL_DIR / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_test_comparison() -> None:
    rows = read_csv(FINAL_DIR / "test_comparison.csv")
    methods = [row["method"] for row in rows]
    metrics = ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]

    x = list(range(len(metrics)))
    width = 0.24

    plt.figure(figsize=(10, 5.5))

    for i, row in enumerate(rows):
        values = [float(row[metric]) for metric in metrics]
        positions = [value + (i - 1) * width for value in x]
        plt.bar(positions, values, width=width, label=methods[i])

    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Final test retrieval quality")
    plt.legend()
    save_figure("test_comparison")


def plot_dense_nprobe() -> None:
    rows = read_csv(FINAL_DIR / "dense_nprobe_ablation.csv")
    nprobe = [int(row["nprobe"]) for row in rows]

    plt.figure(figsize=(9, 5.5))

    for metric in ("Recall@10", "MRR@10", "nDCG@10"):
        values = [float(row[metric]) for row in rows]
        plt.plot(nprobe, values, marker="o", label=metric)

    plt.xlabel("nprobe")
    plt.ylabel("Score")
    plt.title("Dense Retrieval nprobe ablation on dev")
    plt.xticks(nprobe)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("dense_nprobe_ablation")


def plot_hybrid_weight() -> None:
    rows = read_csv(FINAL_DIR / "hybrid_weight_ablation.csv")
    rows = sorted(rows, key=lambda row: float(row["bm25_weight"]))

    weights = [float(row["bm25_weight"]) for row in rows]

    plt.figure(figsize=(9, 5.5))

    for metric in ("Recall@10", "MRR@10", "nDCG@10"):
        values = [float(row[metric]) for row in rows]
        plt.plot(weights, values, marker="o", label=metric)

    plt.xlabel("BM25 weight")
    plt.ylabel("Score")
    plt.title("Weighted RRF ablation on dev")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("hybrid_weight_ablation")


def plot_latency() -> None:
    rows = read_csv(FINAL_DIR / "latency_summary.csv")
    methods = [row["method"] for row in rows]
    values = [float(row["average_search_ms_per_query"]) for row in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, values)

    plt.ylabel("Average search latency, ms/query")
    plt.title("Test retrieval latency")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    save_figure("latency_test")


def plot_error_analysis() -> None:
    rows = read_csv(FINAL_DIR / "error_analysis_summary.csv")
    rows = [row for row in rows if row["section"] == "category_counts"]

    label_map = {
        "all_success": "All succeed",
        "all_fail": "All fail",
        "bm25_success_dense_fail": "BM25 success / Dense fail",
        "dense_success_bm25_fail": "Dense success / BM25 fail",
        "dense_success_hybrid_fail": "Dense success / Hybrid fail",
    }

    labels = [label_map.get(row["name"], row["name"]) for row in rows]
    counts = [int(row["count"]) for row in rows]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(labels, counts)

    plt.ylabel("Queries")
    plt.title("Test error-analysis categories")
    plt.xticks(rotation=18, ha="right")

    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    save_figure("error_analysis_categories")


def main() -> None:
    plot_test_comparison()
    plot_dense_nprobe()
    plot_hybrid_weight()
    plot_latency()
    plot_error_analysis()

    print("=== Final figures ===")
    for path in sorted(FIGURES_DIR.glob("*")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
