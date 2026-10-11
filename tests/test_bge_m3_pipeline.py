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

import build_bge_m3_index  # noqa: E402
import run_bge_m3  # noqa: E402
from bge_m3_utils import (  # noqa: E402
    BGE_INDEX_DIR,
    DEFAULT_RUN_PATH,
    DEFAULT_TIMING_PATH,
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    document_text,
    query_text,
    validate_index_artifacts,
)
from dense_utils import DENSE_INDEX_DIR  # noqa: E402


class _FakeBgeModel:
    encoded_batches: list[tuple[list[str], dict[str, object]]] = []
    instances: list["_FakeBgeModel"] = []
    dimension = EMBEDDING_DIMENSION

    def __init__(self, _model: str, *, device: str) -> None:
        self.device = device
        self.max_seq_length: int | None = None
        self.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encoded_batches.append((list(texts), kwargs))
        return np.ones((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)


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


class BgeM3PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_corpus(
        self,
        corpus_path: Path,
        *,
        count: int = 10,
        docids: list[str] | None = None,
    ) -> None:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        docids = docids or [f"doc-{number}" for number in range(count)]
        with gzip.open(corpus_path, "wt", encoding="utf-8") as stream:
            for number, docid in enumerate(docids):
                json.dump(
                    {
                        "docid": docid,
                        "title": f"title-{number}",
                        "text": f"text-{number}",
                    },
                    stream,
                )
                stream.write("\n")

    def build_argv(self, corpus_path: Path, index_dir: Path, *extra: str) -> list[str]:
        return [
            "build_bge_m3_index.py",
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
            "2",
            "--device",
            "cpu",
            *extra,
        ]

    def write_complete_artifacts(
        self,
        index_dir: Path,
        *,
        docids: list[str] | None,
        document_count: int = 2,
        expected_document_count: int = 2,
    ) -> None:
        index_dir.mkdir(parents=True)
        for name in ("index.faiss", "build_timing.json", ".complete"):
            (index_dir / name).write_text("ok\n", encoding="utf-8")
        if docids is not None:
            (index_dir / "docids.txt").write_text(
                "".join(f"{docid}\n" for docid in docids), encoding="utf-8"
            )
        (index_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "document_count": document_count,
                    "expected_document_count": expected_document_count,
                }
            ),
            encoding="utf-8",
        )

    def test_document_and_query_formatting_have_no_e5_prefixes(self) -> None:
        self.assertEqual(document_text("title", "text"), "title text")
        self.assertEqual(query_text("вопрос"), "вопрос")
        self.assertFalse(document_text("title", "text").startswith("passage: "))
        self.assertFalse(query_text("вопрос").startswith("query: "))

    def test_build_uses_frozen_bge_settings_and_custom_paths(self) -> None:
        corpus_path = self.root / "data" / "corpus.jsonl.gz"
        index_dir = self.root / "indexes" / "dense-bge-m3"
        self.write_corpus(corpus_path)
        digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        _FakeBgeModel.encoded_batches = []
        _FakeBgeModel.instances = []
        with patch.object(
            sys,
            "argv",
            self.build_argv(corpus_path, index_dir, "--corpus-sha256", digest.upper()),
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", _FakeBgeModel
        ), patch.object(build_bge_m3_index, "faiss", _FakeBuildFaiss):
            build_bge_m3_index.main()

        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model"], MODEL_NAME)
        self.assertEqual(metadata["corpus_path"], str(corpus_path.resolve()))
        self.assertEqual(metadata["corpus_sha256"], digest)
        self.assertEqual(metadata["document_count"], 10)
        self.assertEqual(metadata["expected_document_count"], 10)
        self.assertEqual(metadata["max_sequence_length"], 512)
        self.assertEqual(metadata["embedding_dimension"], 1024)
        self.assertTrue(metadata["normalized_embeddings"])
        self.assertEqual(metadata["document_format"], "title + ' ' + text")
        self.assertEqual(metadata["query_format"], "raw query text")
        self.assertEqual(metadata["metric"], "inner_product")
        self.assertEqual(metadata["faiss_type"], "IVFScalarQuantizer(SQ8, inner-product)")
        self.assertNotEqual(index_dir, DENSE_INDEX_DIR)
        self.assertTrue((index_dir / ".complete").is_file())
        self.assertEqual(_FakeBgeModel.instances[0].max_seq_length, 512)
        self.assertTrue(
            all(
                kwargs["normalize_embeddings"] is True
                and not any(text.startswith("passage: ") for text in texts)
                for texts, kwargs in _FakeBgeModel.encoded_batches
            )
        )

    def test_sha_failures_precede_model_and_faiss_work(self) -> None:
        corpus_path = self.root / "data" / "corpus.jsonl.gz"
        self.write_corpus(corpus_path)
        for label, digest in (("incorrect", "0" * 64), ("malformed", "not-a-sha")):
            with self.subTest(kind=label):
                model = Mock()
                index_flat = Mock()
                with patch.object(
                    sys,
                    "argv",
                    self.build_argv(
                        corpus_path,
                        self.root / "indexes" / label,
                        "--corpus-sha256",
                        digest,
                    ),
                ), patch.object(
                    build_bge_m3_index, "SentenceTransformer", model
                ), patch.object(build_bge_m3_index.faiss, "IndexFlatIP", index_flat):
                    with self.assertRaisesRegex(ValueError, "SHA-256|hex"):
                        build_bge_m3_index.main()
                model.assert_not_called()
                index_flat.assert_not_called()

    def test_dimension_mismatch_fails_before_faiss_construction(self) -> None:
        corpus_path = self.root / "data" / "corpus.jsonl.gz"
        self.write_corpus(corpus_path)

        class WrongDimensionModel(_FakeBgeModel):
            dimension = 384

        index_flat = Mock()
        with patch.object(
            sys, "argv", self.build_argv(corpus_path, self.root / "indexes" / "bad")
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", WrongDimensionModel
        ), patch.object(build_bge_m3_index.faiss, "IndexFlatIP", index_flat):
            with self.assertRaisesRegex(RuntimeError, "embedding dimension"):
                build_bge_m3_index.main()
        index_flat.assert_not_called()

    def test_build_rejects_corpus_larger_than_expected_count(self) -> None:
        corpus_path = self.root / "data" / "corpus.jsonl.gz"
        self.write_corpus(corpus_path, count=11)
        with patch.object(
            sys,
            "argv",
            self.build_argv(corpus_path, self.root / "indexes" / "too-many"),
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", _FakeBgeModel
        ), patch.object(build_bge_m3_index, "faiss", _FakeBuildFaiss):
            with self.assertRaisesRegex(RuntimeError, "больше документов"):
                build_bge_m3_index.main()

    def test_duplicate_docid_across_batches_is_rejected(self) -> None:
        corpus_path = self.root / "data" / "across-batches.jsonl.gz"
        self.write_corpus(
            corpus_path,
            docids=["duplicate", "doc-1", "duplicate", *[f"doc-{n}" for n in range(3, 10)]],
        )
        with patch.object(
            sys,
            "argv",
            self.build_argv(corpus_path, self.root / "indexes" / "across-batches"),
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", _FakeBgeModel
        ), patch.object(build_bge_m3_index, "faiss", _FakeBuildFaiss):
            with self.assertRaisesRegex(RuntimeError, "Повтор docid.*duplicate"):
                build_bge_m3_index.main()

    def test_duplicate_docid_inside_batch_is_rejected(self) -> None:
        corpus_path = self.root / "data" / "inside-batch.jsonl.gz"
        self.write_corpus(
            corpus_path,
            docids=["duplicate", "duplicate", *[f"doc-{n}" for n in range(2, 10)]],
        )
        with patch.object(
            sys,
            "argv",
            self.build_argv(corpus_path, self.root / "indexes" / "inside-batch"),
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", _FakeBgeModel
        ), patch.object(build_bge_m3_index, "faiss", _FakeBuildFaiss):
            with self.assertRaisesRegex(RuntimeError, "Повтор docid.*duplicate"):
                build_bge_m3_index.main()

    def test_unique_docids_are_accepted(self) -> None:
        corpus_path = self.root / "data" / "unique.jsonl.gz"
        index_dir = self.root / "indexes" / "unique"
        self.write_corpus(corpus_path)
        with patch.object(
            sys, "argv", self.build_argv(corpus_path, index_dir)
        ), patch.object(
            build_bge_m3_index, "SentenceTransformer", _FakeBgeModel
        ), patch.object(build_bge_m3_index, "faiss", _FakeBuildFaiss):
            build_bge_m3_index.main()
        self.assertTrue((index_dir / ".complete").is_file())

    def test_incomplete_index_cannot_be_used_for_retrieval(self) -> None:
        index_dir = self.root / "indexes" / "incomplete"
        index_dir.mkdir(parents=True)
        with patch.object(
            sys, "argv", ["run_bge_m3.py", "--index-dir", str(index_dir)]
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Отсутствует BGE-M3 artifact"):
                run_bge_m3.main()

    def test_complete_artifacts_with_matching_docids_are_accepted(self) -> None:
        index_dir = self.root / "indexes" / "complete"
        self.write_complete_artifacts(index_dir, docids=["doc-1", "doc-2"])
        metadata = validate_index_artifacts(index_dir, index_ntotal=2)
        self.assertEqual(metadata["document_count"], 2)

    def test_missing_docids_artifact_is_rejected(self) -> None:
        index_dir = self.root / "indexes" / "missing-docids"
        self.write_complete_artifacts(index_dir, docids=None)
        with self.assertRaisesRegex(FileNotFoundError, "docids.txt"):
            validate_index_artifacts(index_dir, index_ntotal=2)

    def test_empty_docids_artifact_is_rejected(self) -> None:
        index_dir = self.root / "indexes" / "empty-docids"
        self.write_complete_artifacts(index_dir, docids=[])
        with self.assertRaisesRegex(RuntimeError, "Пустой BGE-M3 artifact.*docids.txt"):
            validate_index_artifacts(index_dir, index_ntotal=2)

    def test_too_few_docids_are_rejected(self) -> None:
        index_dir = self.root / "indexes" / "few-docids"
        self.write_complete_artifacts(index_dir, docids=["doc-1"])
        with self.assertRaisesRegex(RuntimeError, "Число docid"):
            validate_index_artifacts(index_dir, index_ntotal=2)

    def test_too_many_docids_are_rejected(self) -> None:
        index_dir = self.root / "indexes" / "many-docids"
        self.write_complete_artifacts(index_dir, docids=["doc-1", "doc-2", "doc-3"])
        with self.assertRaisesRegex(RuntimeError, "Число docid"):
            validate_index_artifacts(index_dir, index_ntotal=2)

    def test_faiss_count_mismatch_is_rejected(self) -> None:
        index_dir = self.root / "indexes" / "faiss-mismatch"
        self.write_complete_artifacts(index_dir, docids=["doc-1", "doc-2"])
        with self.assertRaisesRegex(RuntimeError, "FAISS ntotal"):
            validate_index_artifacts(index_dir, index_ntotal=3)

    def test_retrieval_uses_raw_queries_and_custom_output_paths(self) -> None:
        index_dir = self.root / "indexes" / "dense-bge-m3"
        index_dir.mkdir(parents=True)
        for name in ("index.faiss", "build_timing.json", ".complete"):
            (index_dir / name).write_text("ok\n", encoding="utf-8")
        (index_dir / "docids.txt").write_text("doc-1\n", encoding="utf-8")
        metadata = {
            "model": MODEL_NAME,
            "embedding_dimension": 1024,
            "max_sequence_length": 512,
            "normalized_embeddings": True,
            "document_format": "title + ' ' + text",
            "query_format": "raw query text",
            "retrieval_mode": "dense",
            "metric": "inner_product",
            "faiss_type": "IVFScalarQuantizer(SQ8, inner-product)",
            "document_count": 1,
            "expected_document_count": 1,
            "nlist": 1,
            "train_docs": 10,
        }
        (index_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        run_path = self.root / "results" / "runs" / "dense_bge_m3_dev.trec"
        timing_path = self.root / "results" / "tables" / "dense_bge_m3_dev_timing.json"
        fake_faiss = _FakeSearchFaiss()
        _FakeBgeModel.encoded_batches = []
        _FakeBgeModel.instances = []
        with patch.object(
            sys,
            "argv",
            [
                "run_bge_m3.py",
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
        ), patch.object(run_bge_m3, "faiss", fake_faiss), patch.object(
            run_bge_m3, "SentenceTransformer", _FakeBgeModel
        ), patch.object(run_bge_m3, "split_paths", return_value=(Path("topics.dev.txt"), Path("qrels.dev.txt"))), patch.object(
            run_bge_m3, "load_topics", return_value=[("q1", "сырой запрос")]
        ):
            run_bge_m3.main()
        self.assertIn("q1 Q0 doc-1 1", run_path.read_text(encoding="utf-8"))
        self.assertEqual(fake_faiss.index.nprobe, 256)
        self.assertEqual(_FakeBgeModel.encoded_batches[0][0], ["сырой запрос"])
        self.assertTrue(_FakeBgeModel.encoded_batches[0][1]["normalize_embeddings"])
        self.assertEqual(_FakeBgeModel.instances[0].max_seq_length, 512)
        self.assertEqual(json.loads(timing_path.read_text(encoding="utf-8"))["nprobe"], 256)

    def test_default_artifact_paths_are_isolated_from_e5(self) -> None:
        with patch.object(sys, "argv", ["build_bge_m3_index.py"]):
            build_defaults = build_bge_m3_index.parse_args()
        with patch.object(sys, "argv", ["run_bge_m3.py"]):
            run_defaults = run_bge_m3.parse_args()
        self.assertEqual(build_defaults.index_dir, BGE_INDEX_DIR)
        self.assertEqual(run_defaults.run_path, DEFAULT_RUN_PATH)
        self.assertEqual(run_defaults.timing_path, DEFAULT_TIMING_PATH)
        self.assertNotEqual(build_defaults.index_dir, DENSE_INDEX_DIR)


if __name__ == "__main__":
    unittest.main()
