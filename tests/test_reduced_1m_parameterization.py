from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_bm25_index  # noqa: E402
import build_dense_index  # noqa: E402
import run_bm25  # noqa: E402
import run_dense  # noqa: E402
from data_utils import CORPUS_PATH, INDEX_DIR, SUPPORTED_SPLITS  # noqa: E402
from dense_utils import (  # noqa: E402
    DENSE_INDEX_DIR,
    FULL_CORPUS_DOCUMENTS,
    MODEL_NAME,
    passage_text,
    query_text,
)


class _FakeDenseModel:
    encoded_batches: list[list[str]] = []

    def __init__(self, _model: str, *, device: str) -> None:
        self.device = device

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        self.encoded_batches.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32)


class _FakeBuildIndex:
    def __init__(self) -> None:
        self.is_trained = False
        self.ntotal = 0

    def train(self, _vectors: np.ndarray) -> None:
        self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        self.ntotal += len(vectors)


class _FakeBuildFaiss:
    __version__ = "fake"
    METRIC_INNER_PRODUCT = 0

    class ScalarQuantizer:
        QT_8bit = 0

    @staticmethod
    def IndexFlatIP(_dimension: int) -> object:
        return object()

    @staticmethod
    def IndexIVFScalarQuantizer(*_args: object) -> _FakeBuildIndex:
        return _FakeBuildIndex()

    @staticmethod
    def write_index(_index: _FakeBuildIndex, path: str) -> None:
        Path(path).write_bytes(b"fake-faiss-index")


class _FakeSearchIndex:
    ntotal = 1
    nprobe = 0

    def search(self, vectors: np.ndarray, _top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((len(vectors), 1), 0.5, dtype=np.float32),
            np.zeros((len(vectors), 1), dtype=np.int64),
        )


class _FakeSearchFaiss:
    def __init__(self) -> None:
        self.index = _FakeSearchIndex()

    def read_index(self, _path: str) -> _FakeSearchIndex:
        return self.index


class ReducedOneMillionParameterizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_dense_corpus(self, corpus_path: Path, *, count: int = 10) -> None:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(corpus_path, "wt", encoding="utf-8") as stream:
            for number in range(count):
                json.dump(
                    {"docid": f"doc-{number}", "title": "title", "text": "text"},
                    stream,
                )
                stream.write("\n")

    def dense_build_argv(
        self,
        corpus_path: Path,
        index_dir: Path,
        *extra_args: str,
    ) -> list[str]:
        return [
            "build_dense_index.py",
            "--corpus-path",
            str(corpus_path),
            "--index-dir",
            str(index_dir),
            "--expected-document-count",
            "10",
            "--nlist",
            "1",
            "--train-docs",
            "10",
            "--batch-size",
            "5",
            "--device",
            "cpu",
            *extra_args,
        ]

    def test_bm25_custom_paths_are_forwarded_and_defaults_are_historical(self) -> None:
        with patch.object(sys, "argv", ["build_bm25_index.py"]):
            defaults = build_bm25_index.parse_args()
        self.assertEqual(defaults.corpus_path, CORPUS_PATH)
        self.assertEqual(defaults.index_dir, INDEX_DIR)

        corpus_path = self.root / "reduced" / "corpus.jsonl.gz"
        index_dir = self.root / "indexes" / "bm25"
        with patch.object(
            sys,
            "argv",
            [
                "build_bm25_index.py",
                "--corpus-path",
                str(corpus_path),
                "--index-dir",
                str(index_dir),
            ],
        ), patch.object(build_bm25_index, "build_index") as build:
            build_bm25_index.main()

        build.assert_called_once_with(
            corpus_path=corpus_path,
            index_dir=index_dir,
            force=False,
            max_docs=None,
            required_qrels_splits=None,
        )

    def test_custom_bm25_corpus_defaults_to_dev_and_test_qrels(self) -> None:
        reduced_corpus_path = self.root / "reduced" / "corpus.jsonl.gz"
        self.write_dense_corpus(reduced_corpus_path, count=1)
        self.assertEqual(
            build_bm25_index.default_required_qrels_splits(CORPUS_PATH),
            SUPPORTED_SPLITS,
        )
        self.assertEqual(
            build_bm25_index.default_required_qrels_splits(reduced_corpus_path),
            ("dev", "test"),
        )
        with patch.object(
            build_bm25_index,
            "referenced_relevant_docids",
            return_value={"doc-0"},
        ) as referenced:
            build_bm25_index.read_corpus(
                reduced_corpus_path,
                required_qrels_splits=build_bm25_index.default_required_qrels_splits(
                    reduced_corpus_path
                ),
            )
        referenced.assert_called_once_with(("dev", "test"))

    def test_bm25_custom_run_and_timing_paths_are_forwarded(self) -> None:
        with patch.object(sys, "argv", ["run_bm25.py"]):
            defaults = run_bm25.parse_args()
        self.assertEqual(defaults.index_dir, INDEX_DIR)
        self.assertIsNone(defaults.run_path)
        self.assertIsNone(defaults.timing_path)

        index_dir = self.root / "indexes" / "bm25"
        run_path = self.root / "results" / "runs" / "bm25_dev.trec"
        timing_path = self.root / "results" / "tables" / "bm25_dev_timing.json"
        with patch.object(
            sys,
            "argv",
            [
                "run_bm25.py",
                "--index-dir",
                str(index_dir),
                "--run-path",
                str(run_path),
                "--timing-path",
                str(timing_path),
            ],
        ), patch.object(run_bm25, "run_search") as run_search:
            run_bm25.main()

        run_search.assert_called_once_with(
            split="dev",
            top_k=100,
            index_dir=index_dir,
            run_path=run_path,
            timing_path=timing_path,
        )

    def test_dense_build_honors_custom_corpus_and_index_and_records_metadata(self) -> None:
        corpus_path = self.root / "reduced" / "corpus.jsonl.gz"
        self.write_dense_corpus(corpus_path)

        index_dir = self.root / "indexes" / "dense-e5-small"
        _FakeDenseModel.encoded_batches = []
        digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        with patch.object(
            sys,
            "argv",
            self.dense_build_argv(
                corpus_path,
                index_dir,
                "--corpus-sha256",
                digest.upper(),
            ),
        ), patch.object(build_dense_index, "SentenceTransformer", _FakeDenseModel), patch.object(
            build_dense_index, "faiss", _FakeBuildFaiss
        ):
            build_dense_index.main()

        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["corpus_path"], str(corpus_path.resolve()))
        self.assertEqual(metadata["corpus_sha256"], digest)
        self.assertEqual(metadata["document_count"], 10)
        self.assertEqual(metadata["expected_document_count"], 10)
        self.assertEqual(metadata["model"], MODEL_NAME)
        self.assertEqual(metadata["embedding_dimension"], 2)
        self.assertTrue(metadata["normalized_embeddings"])
        self.assertEqual(metadata["document_prefix"], "passage: ")
        self.assertEqual(metadata["query_prefix"], "query: ")
        self.assertEqual(metadata["faiss_type"], "IVFScalarQuantizer(SQ8, inner-product)")
        self.assertEqual(metadata["nlist"], 1)
        self.assertNotEqual(index_dir, DENSE_INDEX_DIR)
        self.assertTrue(
            all(
                text.startswith("passage: ")
                for batch in _FakeDenseModel.encoded_batches
                for text in batch
            )
        )

    def test_dense_build_rejects_invalid_sha_before_model_or_faiss_construction(self) -> None:
        corpus_path = self.root / "reduced" / "corpus.jsonl.gz"
        self.write_dense_corpus(corpus_path)

        for label, digest in (("incorrect", "0" * 64), ("malformed", "not-a-sha")):
            with self.subTest(kind=label):
                model = Mock()
                index_flat = Mock()
                with patch.object(
                    sys,
                    "argv",
                    self.dense_build_argv(
                        corpus_path,
                        self.root / "indexes" / label,
                        "--corpus-sha256",
                        digest,
                    ),
                ), patch.object(
                    build_dense_index, "SentenceTransformer", model
                ), patch.object(build_dense_index.faiss, "IndexFlatIP", index_flat):
                    with self.assertRaisesRegex(ValueError, "SHA-256|hex"):
                        build_dense_index.main()
                model.assert_not_called()
                index_flat.assert_not_called()

    def test_dense_build_without_sha_skips_hashing_and_remains_valid(self) -> None:
        corpus_path = self.root / "reduced" / "corpus.jsonl.gz"
        self.write_dense_corpus(corpus_path)
        index_dir = self.root / "indexes" / "dense-e5-small"
        _FakeDenseModel.encoded_batches = []
        with patch.object(
            sys, "argv", self.dense_build_argv(corpus_path, index_dir)
        ), patch.object(build_dense_index, "SentenceTransformer", _FakeDenseModel), patch.object(
            build_dense_index, "faiss", _FakeBuildFaiss
        ), patch.object(
            build_dense_index.hashlib,
            "sha256",
            side_effect=AssertionError("hashing must be skipped when no digest is supplied"),
        ):
            build_dense_index.main()

        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertIsNone(metadata["corpus_sha256"])

    def test_dense_retrieval_honors_custom_index_run_and_timing_paths(self) -> None:
        index_dir = self.root / "indexes" / "dense-e5-small"
        index_dir.mkdir(parents=True)
        for name in ("index.faiss", ".complete"):
            (index_dir / name).write_text("ok\n", encoding="utf-8")
        (index_dir / "docids.txt").write_text("doc-1\n", encoding="utf-8")
        (index_dir / "metadata.json").write_text(
            json.dumps({"is_subset": False, "model": MODEL_NAME}), encoding="utf-8"
        )
        run_path = self.root / "results" / "runs" / "dense_e5_small_dev.trec"
        timing_path = self.root / "results" / "tables" / "dense_e5_small_dev_timing.json"
        fake_faiss = _FakeSearchFaiss()
        _FakeDenseModel.encoded_batches = []

        with patch.object(
            sys,
            "argv",
            [
                "run_dense.py",
                "--index-dir",
                str(index_dir),
                "--run-path",
                str(run_path),
                "--timing-path",
                str(timing_path),
                "--nprobe",
                "256",
                "--device",
                "cpu",
            ],
        ), patch.object(run_dense, "faiss", fake_faiss), patch.object(
            run_dense, "SentenceTransformer", _FakeDenseModel
        ), patch.object(run_dense, "load_topics", return_value=[("q1", "query: пример")]):
            run_dense.main()

        self.assertTrue(run_path.is_file())
        self.assertTrue(timing_path.is_file())
        self.assertIn("q1 Q0 doc-1 1", run_path.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(timing_path.read_text(encoding="utf-8"))["nprobe"], 256)
        self.assertEqual(fake_faiss.index.nprobe, 256)
        self.assertEqual(_FakeDenseModel.encoded_batches, [["query: пример"]])

    def test_e5_prefixes_and_historical_dense_defaults_remain_unchanged(self) -> None:
        self.assertEqual(passage_text("title", "text"), "passage: title text")
        self.assertEqual(query_text("text"), "query: text")
        with patch.object(sys, "argv", ["build_dense_index.py"]):
            build_defaults = build_dense_index.parse_args()
        with patch.object(sys, "argv", ["run_dense.py"]):
            run_defaults = run_dense.parse_args()
        self.assertEqual(build_defaults.corpus_path, CORPUS_PATH)
        self.assertEqual(build_defaults.index_dir, DENSE_INDEX_DIR)
        self.assertEqual(build_defaults.expected_document_count, FULL_CORPUS_DOCUMENTS)
        self.assertEqual(build_defaults.nlist, 4096)
        self.assertEqual(run_defaults.index_dir, DENSE_INDEX_DIR)
        self.assertEqual(run_defaults.top_k, 100)


if __name__ == "__main__":
    unittest.main()
