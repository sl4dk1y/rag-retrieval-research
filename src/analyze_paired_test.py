"""Paired statistical analysis for the frozen reduced-1M final TEST runs.

This module deliberately evaluates the already-written TREC runs only.  It
does not retrieve documents, build an index, or change an existing experiment
artifact.  Confidence intervals use paired bootstrap resampling; p-values use
the separately specified paired sign-flip randomization test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from data_utils import (
    PROJECT_ROOT,
    DataFormatError,
    load_json,
    load_qrels,
    load_topics,
    validate_split_files,
    write_json,
)
from evaluate_run import (
    METRIC_CUTOFFS,
    calculate_metrics,
    load_trec_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


SPLIT = "test"
EXPECTED_TEST_QUERY_COUNT = 995
SEED = 42
BOOTSTRAP_ITERATIONS = 10_000
RANDOMIZATION_ITERATIONS = 10_000
CONFIDENCE_LEVEL = 0.95
AGGREGATE_TOLERANCE = 1e-12
COMPARISON_TOLERANCE = 1e-12
P_VALUE_TOLERANCE = 1e-15
METRIC_NAMES = tuple(f"Recall@{cutoff}" for cutoff in METRIC_CUTOFFS) + (
    "MRR@10",
    "nDCG@10",
)

REDUCED_RESULTS_DIR = PROJECT_ROOT / "results" / "reduced-1m"
RUN_SPECS: dict[str, tuple[Path, Path]] = {
    "e5": (
        REDUCED_RESULTS_DIR / "runs" / "dense_e5_small_test.trec",
        REDUCED_RESULTS_DIR / "tables" / "dense_e5_small_test_metrics.json",
    ),
    "e5_bm25_rrf": (
        REDUCED_RESULTS_DIR / "runs" / "hybrid_rrf_test.trec",
        REDUCED_RESULTS_DIR / "tables" / "hybrid_rrf_test_metrics.json",
    ),
    "bge_m3": (
        REDUCED_RESULTS_DIR / "runs" / "dense_bge_m3_test.trec",
        REDUCED_RESULTS_DIR / "tables" / "dense_bge_m3_test_metrics.json",
    ),
    "bge_m3_bm25_rrf": (
        REDUCED_RESULTS_DIR / "runs" / "hybrid_bge_m3_rrf_test.trec",
        REDUCED_RESULTS_DIR / "tables" / "hybrid_bge_m3_rrf_test_metrics.json",
    ),
}
COMPARISONS = (
    ("bge_m3_vs_e5", "bge_m3", "e5"),
    ("e5_bm25_rrf_vs_e5", "e5_bm25_rrf", "e5"),
    ("bge_m3_bm25_rrf_vs_bge_m3", "bge_m3_bm25_rrf", "bge_m3"),
)
DEFAULT_OUTPUT_PATH = REDUCED_RESULTS_DIR / "tables" / "paired_test_statistics.json"


def _validate_single_query_blocks(path: Path) -> None:
    """Reject a query whose TREC records occur in multiple non-adjacent blocks."""
    completed: set[str] = set()
    current_query_id: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            query_id = parts[0]
            if query_id != current_query_id:
                if current_query_id is not None:
                    completed.add(current_query_id)
                if query_id in completed:
                    raise DataFormatError(
                        "Повторный блок query_id в TREC run "
                        f"{path}:{line_number}: {query_id!r}"
                    )
                current_query_id = query_id


def load_and_validate_run(
    path: Path, expected_query_ids: list[str]
) -> dict[str, list[tuple[str, float]]]:
    """Reuse evaluator parsing and require precisely the expected query set."""
    _validate_single_query_blocks(path)
    run = load_trec_run(path)
    expected = set(expected_query_ids)
    actual = set(run)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise DataFormatError(
            "Query ID в run не совпадают с TEST topics: "
            f"пропущено={len(missing):,}, лишних={len(extra):,}"
        )
    return run


def per_query_metrics(
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
) -> dict[str, np.ndarray]:
    """Calculate evaluator-identical metric values for every query in order."""
    values = {name: np.empty(len(query_ids), dtype=np.float64) for name in METRIC_NAMES}
    for position, query_id in enumerate(query_ids):
        ranking = [doc_id for doc_id, _score in run[query_id]]
        relevances = qrels[query_id]
        for cutoff in METRIC_CUTOFFS:
            values[f"Recall@{cutoff}"][position] = recall_at_k(
                ranking, relevances, cutoff
            )
        values["MRR@10"][position] = reciprocal_rank_at_k(ranking, relevances, 10)
        values["nDCG@10"][position] = ndcg_at_k(ranking, relevances, 10)
    return values


def verify_aggregate_metrics(
    values: dict[str, np.ndarray], expected_metrics: dict[str, Any], method: str
) -> dict[str, float]:
    """Guard against per-query metric drift from the frozen aggregate result."""
    means = {name: float(np.mean(metric_values)) for name, metric_values in values.items()}
    for name in METRIC_NAMES:
        expected = expected_metrics.get(name)
        if not isinstance(expected, (int, float)) or not math.isfinite(expected):
            raise DataFormatError(f"Некорректная агрегированная метрика {name} для {method}")
        if not math.isclose(
            means[name], float(expected), rel_tol=0.0, abs_tol=AGGREGATE_TOLERANCE
        ):
            raise DataFormatError(
                f"Per-query {name} для {method} не совпадает с frozen aggregate: "
                f"{means[name]:.17g} != {float(expected):.17g}"
            )
    return means


def _bootstrap_interval(
    differences: np.ndarray, *, iterations: int, seed: int
) -> tuple[float, float]:
    """Return the two-sided percentile CI for a mean paired difference."""
    rng = np.random.default_rng(seed)
    sample_means = np.empty(iterations, dtype=np.float64)
    batch_size = 250
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        sample_means[start : start + count] = differences[indices].mean(axis=1)
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(sample_means, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _sign_flip_p_value(
    differences: np.ndarray, *, iterations: int, seed: int
) -> float:
    """Monte-Carlo two-sided paired sign-flip p-value with +1 correction."""
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    extreme_count = 0
    batch_size = 250
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        signs = rng.integers(0, 2, size=(count, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        null_means = (signs * differences).mean(axis=1)
        extreme_count += int(
            np.count_nonzero(np.abs(null_means) >= observed - P_VALUE_TOLERANCE)
        )
    return (extreme_count + 1) / (iterations + 1)


def paired_statistics(
    method_a_values: np.ndarray,
    method_b_values: np.ndarray,
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    randomization_iterations: int = RANDOMIZATION_ITERATIONS,
    seed: int = SEED,
) -> dict[str, float | int]:
    """Summarize one metric with fixed paired bootstrap and sign-flip procedures."""
    if method_a_values.shape != method_b_values.shape or method_a_values.ndim != 1:
        raise ValueError("Paired metric arrays must be one-dimensional and equally sized")
    if len(method_a_values) == 0:
        raise ValueError("Paired metric arrays must not be empty")
    differences = method_a_values - method_b_values
    ci_lower, ci_upper = _bootstrap_interval(
        differences, iterations=bootstrap_iterations, seed=seed
    )
    tied = np.isclose(differences, 0.0, rtol=0.0, atol=COMPARISON_TOLERANCE)
    return {
        "observed_difference": float(np.mean(differences)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": _sign_flip_p_value(
            differences, iterations=randomization_iterations, seed=seed
        ),
        "wins": int(np.count_nonzero(differences > COMPARISON_TOLERANCE)),
        "ties": int(np.count_nonzero(tied)),
        "losses": int(np.count_nonzero(differences < -COMPARISON_TOLERANCE)),
    }


def analyze_runs(
    *,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[tuple[str, float]]]],
    expected_aggregates: dict[str, dict[str, Any]],
    comparisons: tuple[tuple[str, str, str], ...] = COMPARISONS,
    seed: int = SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    randomization_iterations: int = RANDOMIZATION_ITERATIONS,
) -> dict[str, Any]:
    """Analyze fully validated runs; exposed separately for synthetic unit tests."""
    expected_query_set = set(query_ids)
    if len(expected_query_set) != len(query_ids):
        raise DataFormatError("Повтор query_id в списке TEST topics")
    if set(qrels) != expected_query_set:
        raise DataFormatError("Query ID в qrels не совпадают в точности с TEST topics")

    values_by_method: dict[str, dict[str, np.ndarray]] = {}
    means_by_method: dict[str, dict[str, float]] = {}
    for method, run in runs.items():
        if set(run) != expected_query_set:
            raise DataFormatError(f"Query ID run {method} не совпадают с TEST topics")
        values = per_query_metrics(query_ids, qrels, run)
        values_by_method[method] = values
        try:
            expected = expected_aggregates[method]
        except KeyError as error:
            raise DataFormatError(f"Нет frozen aggregate metrics для {method}") from error
        means_by_method[method] = verify_aggregate_metrics(values, expected, method)

    rendered_comparisons: dict[str, Any] = {}
    for comparison_id, method_a, method_b in comparisons:
        if method_a not in values_by_method or method_b not in values_by_method:
            raise DataFormatError(f"Неизвестный метод в сравнении {comparison_id}")
        metrics: dict[str, Any] = {}
        for metric in METRIC_NAMES:
            metrics[metric] = {
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": means_by_method[method_a][metric],
                "mean_b": means_by_method[method_b][metric],
                **paired_statistics(
                    values_by_method[method_a][metric],
                    values_by_method[method_b][metric],
                    bootstrap_iterations=bootstrap_iterations,
                    randomization_iterations=randomization_iterations,
                    seed=seed,
                ),
            }
        rendered_comparisons[comparison_id] = {
            "method_a": method_a,
            "method_b": method_b,
            "metrics": metrics,
        }

    return {
        "split": SPLIT,
        "query_count": len(query_ids),
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "randomization_iterations": randomization_iterations,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_ci_method": (
            "Paired percentile bootstrap: resample query indices with replacement "
            "and take the 2.5th and 97.5th percentiles of mean(A - B)."
        ),
        "p_value_method": (
            "Two-sided paired sign-flip randomization test: independently multiply "
            "each per-query A - B by -1 or +1 under H0, count null mean differences "
            "with abs(null_mean) >= abs(observed_mean) - 1e-15, and report "
            "(extreme_count + 1) / (iterations + 1)."
        ),
        "wins_ties_losses_tolerance": COMPARISON_TOLERANCE,
        "comparisons": rendered_comparisons,
    }


def analyze_final_test(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    seed: int = SEED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    randomization_iterations: int = RANDOMIZATION_ITERATIONS,
) -> dict[str, Any]:
    """Run the analysis against the frozen final TEST artifacts and write JSON."""
    topics_path, qrels_path = validate_split_files(SPLIT)
    topics = load_topics(topics_path)
    qrels = load_qrels(qrels_path)
    query_ids = [query_id for query_id, _query in topics]
    if len(query_ids) != EXPECTED_TEST_QUERY_COUNT:
        raise DataFormatError(
            f"Ожидалось {EXPECTED_TEST_QUERY_COUNT} TEST queries, получено {len(query_ids)}"
        )

    runs: dict[str, dict[str, list[tuple[str, float]]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for method, (run_path, metrics_path) in RUN_SPECS.items():
        runs[method] = load_and_validate_run(run_path, query_ids)
        payload = load_json(metrics_path, description=f"frozen aggregate metrics {method}")
        if payload.get("split") != SPLIT or payload.get("query_count") != len(query_ids):
            raise DataFormatError(f"Некорректные split/query_count в frozen metrics для {method}")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise DataFormatError(f"Некорректное поле metrics для {method}")
        aggregates[method] = metrics

    result = analyze_runs(
        query_ids=query_ids,
        qrels=qrels,
        runs=runs,
        expected_aggregates=aggregates,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
        randomization_iterations=randomization_iterations,
    )
    write_json(output_path, result)
    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print the compact, human-readable counterpart of the JSON artifact."""
    print(
        f"Paired TEST statistics: {result['query_count']} queries, "
        f"seed={result['seed']}, bootstrap/randomization="
        f"{result['bootstrap_iterations']}/{result['randomization_iterations']}"
    )
    for comparison_id, comparison in result["comparisons"].items():
        print(f"\n{comparison_id}: {comparison['method_a']} - {comparison['method_b']}")
        for metric, values in comparison["metrics"].items():
            print(
                f"  {metric:<10} diff={values['observed_difference']:+.6f} "
                f"CI=[{values['ci_lower']:+.6f}, {values['ci_upper']:+.6f}] "
                f"p={values['p_value']:.6f} "
                f"W/T/L={values['wins']}/{values['ties']}/{values['losses']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_final_test(output_path=args.output)
    print_summary(result)
    print(f"Statistics saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (DataFormatError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
