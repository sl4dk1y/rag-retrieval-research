from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyze_paired_test as analysis  # noqa: E402
from data_utils import DataFormatError  # noqa: E402
from evaluate_run import calculate_metrics  # noqa: E402


class PairedTestAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query_ids = ["q1", "q2", "q3"]
        self.qrels = {
            "q1": {"d1": 2, "d2": 1},
            "q2": {"d3": 1},
            "q3": {"d5": 1},
        }
        self.good_run = {
            "q1": [("d1", 3.0), ("d2", 2.0), ("d9", 1.0)],
            "q2": [("d3", 3.0), ("d9", 2.0), ("d8", 1.0)],
            "q3": [("d5", 3.0), ("d9", 2.0), ("d8", 1.0)],
        }
        self.bad_run = {
            "q1": [("d9", 3.0), ("d1", 2.0), ("d2", 1.0)],
            "q2": [("d9", 3.0), ("d3", 2.0), ("d8", 1.0)],
            "q3": [("d9", 3.0), ("d5", 2.0), ("d8", 1.0)],
        }

    def aggregates(self, *methods: tuple[str, dict[str, list[tuple[str, float]]]]) -> dict[str, dict[str, float]]:
        return {
            name: calculate_metrics(self.query_ids, self.qrels, run)
            for name, run in methods
        }

    def analyze(
        self, runs: dict[str, dict[str, list[tuple[str, float]]]], comparisons: tuple[tuple[str, str, str], ...]
    ) -> dict[str, object]:
        return analysis.analyze_runs(
            query_ids=self.query_ids,
            qrels=self.qrels,
            runs=runs,
            expected_aggregates=self.aggregates(*runs.items()),
            comparisons=comparisons,
            bootstrap_iterations=200,
            randomization_iterations=200,
        )

    def test_identical_methods_are_all_ties_with_zero_difference(self) -> None:
        result = self.analyze(
            {"a": self.good_run, "b": self.good_run}, (("a_vs_b", "a", "b"),)
        )
        metric = result["comparisons"]["a_vs_b"]["metrics"]["MRR@10"]
        self.assertEqual(metric["observed_difference"], 0.0)
        self.assertEqual(metric["wins"], 0)
        self.assertEqual(metric["ties"], 3)
        self.assertEqual(metric["losses"], 0)
        self.assertEqual(metric["ci_lower"], 0.0)
        self.assertEqual(metric["ci_upper"], 0.0)
        self.assertEqual(metric["p_value"], 1.0)

    def test_superior_and_reversed_comparisons_reverse_signs(self) -> None:
        result = self.analyze(
            {"good": self.good_run, "bad": self.bad_run},
            (("good_vs_bad", "good", "bad"), ("bad_vs_good", "bad", "good")),
        )
        positive = result["comparisons"]["good_vs_bad"]["metrics"]["MRR@10"]
        negative = result["comparisons"]["bad_vs_good"]["metrics"]["MRR@10"]
        self.assertGreater(positive["observed_difference"], 0.0)
        self.assertEqual(positive["wins"], 3)
        self.assertLess(negative["observed_difference"], 0.0)
        self.assertAlmostEqual(
            positive["observed_difference"], -negative["observed_difference"]
        )

    def test_fixed_seed_is_deterministic(self) -> None:
        first = analysis.paired_statistics(
            np.array([0.0, 0.3, 0.7]), np.zeros(3), bootstrap_iterations=200,
            randomization_iterations=200, seed=42
        )
        second = analysis.paired_statistics(
            np.array([0.0, 0.3, 0.7]), np.zeros(3), bootstrap_iterations=200,
            randomization_iterations=200, seed=42
        )
        self.assertEqual(first, second)

    def test_mismatched_query_sets_are_rejected(self) -> None:
        incomplete = dict(self.good_run)
        incomplete.pop("q3")
        with self.assertRaisesRegex(DataFormatError, "не совпадают"):
            analysis.analyze_runs(
                query_ids=self.query_ids,
                qrels=self.qrels,
                runs={"a": self.good_run, "b": incomplete},
                expected_aggregates=self.aggregates(("a", self.good_run)),
                comparisons=(("a_vs_b", "a", "b"),),
                bootstrap_iterations=20,
                randomization_iterations=20,
            )

    def test_duplicate_query_block_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.trec"
            path.write_text(
                "q1 Q0 d1 1 1.0 run\nq2 Q0 d2 1 1.0 run\nq1 Q0 d3 1 1.0 run\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataFormatError, "Повторный блок"):
                analysis.load_and_validate_run(path, ["q1", "q2"])

    def test_malformed_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "malformed.trec"
            path.write_text("q1 Q0 d1 1 not-a-score run\n", encoding="utf-8")
            with self.assertRaisesRegex(DataFormatError, "rank/score"):
                analysis.load_and_validate_run(path, ["q1"])

    def test_per_query_metrics_match_evaluator_aggregate(self) -> None:
        values = analysis.per_query_metrics(self.query_ids, self.qrels, self.good_run)
        expected = calculate_metrics(self.query_ids, self.qrels, self.good_run)
        actual = {name: float(np.mean(value)) for name, value in values.items()}
        self.assertEqual(actual, expected)

    def test_aggregate_consistency_drift_is_rejected(self) -> None:
        values = analysis.per_query_metrics(self.query_ids, self.qrels, self.good_run)
        expected = calculate_metrics(self.query_ids, self.qrels, self.good_run)
        expected["MRR@10"] += 0.01
        with self.assertRaisesRegex(DataFormatError, "не совпадает"):
            analysis.verify_aggregate_metrics(values, expected, "synthetic")

    def test_json_contains_required_metadata_and_comparison_structure(self) -> None:
        result = self.analyze(
            {"a": self.good_run, "b": self.bad_run}, (("a_vs_b", "a", "b"),)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "paired.json"
            analysis.write_json(output, result)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        for key in (
            "split", "query_count", "seed", "bootstrap_iterations",
            "randomization_iterations", "confidence_level", "bootstrap_ci_method",
            "p_value_method", "comparisons",
        ):
            self.assertIn(key, loaded)
        metric = loaded["comparisons"]["a_vs_b"]["metrics"]["Recall@1"]
        for key in (
            "method_a", "method_b", "mean_a", "mean_b", "observed_difference",
            "ci_lower", "ci_upper", "p_value", "wins", "ties", "losses",
        ):
            self.assertIn(key, metric)


if __name__ == "__main__":
    unittest.main()
