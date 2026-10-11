from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_reduced_corpus import (  # noqa: E402
    CorpusBuildError,
    SelectionPaths,
    build_reduced_corpus,
    load_selection_inputs,
)


class ReducedCorpusBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.corpus_path = self.root / "corpus.jsonl.gz"
        self.dev_qrels_path = self.root / "qrels.dev.txt"
        self.test_qrels_path = self.root / "qrels.test.txt"
        self.bm25_dev_run_path = self.root / "bm25_dev.trec"
        self.bm25_test_run_path = self.root / "bm25_test.trec"
        self.e5_dev_run_path = self.root / "e5_dev.trec"
        self.e5_test_run_path = self.root / "e5_test.trec"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_corpus(self, docids: list[str]) -> None:
        with gzip.open(self.corpus_path, "wt", encoding="utf-8") as stream:
            for docid in docids:
                json.dump({"docid": docid, "title": f"title {docid}", "text": f"text {docid}"}, stream)
                stream.write("\n")

    def write_raw_corpus_lines(self, lines: list[str]) -> None:
        with gzip.open(self.corpus_path, "wt", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")

    def write_selection_files(self) -> None:
        self.dev_qrels_path.write_text(
            "dev-positive 0 qrel-dev 1\ndev-negative 0 qrel-negative 0\n", encoding="utf-8"
        )
        self.test_qrels_path.write_text("test-positive 0 qrel-test 2\n", encoding="utf-8")
        run_paths_and_docids = (
            (self.bm25_dev_run_path, ["bm25-dev"]),
            (self.bm25_test_run_path, ["bm25-test"]),
            (self.e5_dev_run_path, ["e5-dev"]),
            (self.e5_test_run_path, ["e5-test"]),
        )
        for path, docids in run_paths_and_docids:
            path.write_text(
                "".join(
                    f"query Q0 {docid} {rank} {1.0 / rank:.4f} frozen-run\n"
                    for rank, docid in enumerate(docids, start=1)
                ),
                encoding="utf-8",
            )

    def selection_paths(self) -> SelectionPaths:
        return SelectionPaths(
            dev_qrels=self.dev_qrels_path,
            test_qrels=self.test_qrels_path,
            bm25_dev_run=self.bm25_dev_run_path,
            bm25_test_run=self.bm25_test_run_path,
            e5_dev_run=self.e5_dev_run_path,
            e5_test_run=self.e5_test_run_path,
        )

    def read_output_docids(self, output_dir: Path) -> list[str]:
        with gzip.open(output_dir / "corpus.jsonl.gz", "rt", encoding="utf-8") as stream:
            return [json.loads(line)["docid"] for line in stream]

    def build(
        self,
        *,
        output_name: str = "out",
        seed: int = 42,
        target_size: int = 12,
        expected_source_count: int | None = None,
    ) -> dict:
        return build_reduced_corpus(
            corpus_path=self.corpus_path,
            output_dir=self.root / output_name,
            selection_paths=self.selection_paths(),
            target_size=target_size,
            random_seed=seed,
            expected_source_count=(
                len(self.read_source_docids())
                if expected_source_count is None
                else expected_source_count
            ),
        )

    def read_source_docids(self) -> list[str]:
        with gzip.open(self.corpus_path, "rt", encoding="utf-8") as stream:
            return [json.loads(line)["docid"] for line in stream if line.strip()]

    def prepare_valid_fixture(self, *, extra_count: int = 20) -> None:
        self.write_selection_files()
        required = ["qrel-dev", "qrel-test", "bm25-dev", "bm25-test", "e5-dev", "e5-test"]
        extras = ["qrel-negative", *[f"extra-{index}" for index in range(extra_count)]]
        self.write_corpus(required + extras)

    def test_positive_qrels_documents_are_retained(self) -> None:
        self.prepare_valid_fixture()
        self.build()
        output_docids = set(self.read_output_docids(self.root / "out"))
        self.assertTrue({"qrel-dev", "qrel-test"}.issubset(output_docids))

    def test_required_run_documents_are_retained(self) -> None:
        self.prepare_valid_fixture()
        self.build()
        output_docids = set(self.read_output_docids(self.root / "out"))
        self.assertTrue({"bm25-dev", "bm25-test", "e5-dev", "e5-test"}.issubset(output_docids))

    def test_output_has_exact_requested_size(self) -> None:
        self.prepare_valid_fixture()
        self.build(target_size=13)
        self.assertEqual(len(self.read_output_docids(self.root / "out")), 13)

    def test_output_docids_are_unique(self) -> None:
        self.prepare_valid_fixture()
        self.build()
        output_docids = self.read_output_docids(self.root / "out")
        self.assertEqual(len(output_docids), len(set(output_docids)))

    def test_same_seed_selects_identical_membership(self) -> None:
        self.prepare_valid_fixture(extra_count=40)
        self.build(output_name="out-one", seed=42)
        self.build(output_name="out-two", seed=42)
        self.assertEqual(
            set(self.read_output_docids(self.root / "out-one")),
            set(self.read_output_docids(self.root / "out-two")),
        )

    def test_different_seed_changes_random_selection_when_candidates_are_plentiful(self) -> None:
        self.prepare_valid_fixture(extra_count=80)
        self.build(output_name="out-one", seed=42)
        self.build(output_name="out-two", seed=99)
        self.assertNotEqual(
            set(self.read_output_docids(self.root / "out-one")),
            set(self.read_output_docids(self.root / "out-two")),
        )

    def test_non_positive_qrels_are_not_required_by_qrels_alone(self) -> None:
        self.prepare_valid_fixture()
        selection_inputs = load_selection_inputs(self.selection_paths(), top_k=100)
        self.assertNotIn("qrel-negative", selection_inputs.required_docids)

    def test_missing_required_source_document_fails(self) -> None:
        self.prepare_valid_fixture()
        self.write_corpus([docid for docid in self.read_source_docids() if docid != "qrel-dev"])
        with self.assertRaisesRegex(CorpusBuildError, "required source docids are missing"):
            self.build()

    def test_malformed_corpus_json_fails(self) -> None:
        self.write_selection_files()
        self.write_raw_corpus_lines(["{not json}"])
        with self.assertRaisesRegex(CorpusBuildError, "Malformed corpus JSON"):
            self.build(target_size=7, expected_source_count=1)

    def test_missing_or_non_string_docid_fails(self) -> None:
        self.write_selection_files()
        for suffix, line in (("missing", '{"title":"t","text":"x"}'), ("number", '{"docid":1,"title":"t","text":"x"}')):
            with self.subTest(kind=suffix):
                self.write_raw_corpus_lines([line])
                with self.assertRaisesRegex(CorpusBuildError, "Missing or non-string docid"):
                    self.build(
                        output_name=f"out-{suffix}", target_size=7, expected_source_count=1
                    )

    def test_duplicate_source_docid_fails(self) -> None:
        self.write_selection_files()
        record = '{"docid":"duplicate","title":"t","text":"x"}'
        self.write_raw_corpus_lines([record, record])
        with self.assertRaisesRegex(CorpusBuildError, "Duplicate source docid"):
            self.build(target_size=7)

    def test_required_union_must_be_smaller_than_target(self) -> None:
        self.prepare_valid_fixture()
        with self.assertRaisesRegex(CorpusBuildError, "must be smaller"):
            self.build(target_size=6)

    def test_manifest_counts_and_coverage_are_correct(self) -> None:
        self.prepare_valid_fixture()
        manifest = self.build(target_size=12)
        self.assertEqual(manifest["required_union_count"], 6)
        self.assertEqual(manifest["randomly_selected_count"], 6)
        self.assertEqual(manifest["final_output_count"], 12)
        self.assertEqual(manifest["positive_dev_qrels_count"], 1)
        self.assertEqual(manifest["positive_test_qrels_count"], 1)
        self.assertEqual(manifest["dev_relevant_document_coverage"], {"covered": 1, "total": 1, "fraction": 1.0})
        self.assertEqual(manifest["total_required_document_coverage"], {"covered": 6, "total": 6, "fraction": 1.0})
        written_manifest = json.loads((self.root / "out" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written_manifest, manifest)

    def test_gzip_bytes_and_sha_are_deterministic(self) -> None:
        self.prepare_valid_fixture(extra_count=40)
        manifest_one = self.build(output_name="out-one")
        manifest_two = self.build(output_name="out-two")
        output_one = (self.root / "out-one" / "corpus.jsonl.gz").read_bytes()
        output_two = (self.root / "out-two" / "corpus.jsonl.gz").read_bytes()
        self.assertEqual(output_one, output_two)
        self.assertEqual(
            hashlib.sha256(output_one).hexdigest(), manifest_one["corpus_jsonl_gz_sha256"]
        )
        self.assertEqual(manifest_one["corpus_jsonl_gz_sha256"], manifest_two["corpus_jsonl_gz_sha256"])


if __name__ == "__main__":
    unittest.main()
