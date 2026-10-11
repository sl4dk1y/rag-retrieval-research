"""Build the frozen one-million-document reduced Russian Mr. TyDi corpus.

The builder deliberately does not read retrieval indexes or BGE-M3 artifacts.  Its
only selection inputs are the frozen BM25/E5 runs and positive DEV/TEST qrels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sqlite3
import tempfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data/raw/mr-tydi-russian/corpus.jsonl.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/mr-tydi-russian-reduced-1m"
DEFAULT_DEV_QRELS_PATH = (
    PROJECT_ROOT / "data/raw/mr-tydi-russian/ir-format-data/qrels.dev.txt"
)
DEFAULT_TEST_QRELS_PATH = (
    PROJECT_ROOT / "data/raw/mr-tydi-russian/ir-format-data/qrels.test.txt"
)
DEFAULT_BM25_DEV_RUN_PATH = PROJECT_ROOT / "results/runs/bm25_dev.trec"
DEFAULT_BM25_TEST_RUN_PATH = PROJECT_ROOT / "results/runs/bm25_test.trec"
DEFAULT_E5_DEV_RUN_PATH = PROJECT_ROOT / "results/runs/dense_e5_small_dev_nprobe256.trec"
DEFAULT_E5_TEST_RUN_PATH = PROJECT_ROOT / "results/runs/dense_e5_small_test_nprobe256.trec"

PROTOCOL_VERSION = "mr-tydi-russian-reduced-corpus-v1"
DEFAULT_TARGET_SIZE = 1_000_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_EXPECTED_SOURCE_COUNT = 9_597_504
DEFAULT_TOP_K = 100


class CorpusBuildError(ValueError):
    """Raised when a source or requested reduced corpus violates the protocol."""


@dataclass(frozen=True)
class SelectionPaths:
    """The frozen qrels and run files that determine required documents."""

    dev_qrels: Path
    test_qrels: Path
    bm25_dev_run: Path
    bm25_test_run: Path
    e5_dev_run: Path
    e5_test_run: Path


@dataclass(frozen=True)
class SelectionInputs:
    """Parsed selection inputs and their pre-union unique-document counts."""

    required_docids: set[str]
    dev_relevant_docids: set[str]
    test_relevant_docids: set[str]
    source_counts: dict[str, int]
    source_paths: dict[str, str]


@dataclass(frozen=True)
class FileSnapshot:
    """Minimal immutable-source check between the two corpus passes."""

    size: int
    mtime_ns: int


def require_nonempty_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if path.stat().st_size == 0:
        raise CorpusBuildError(f"Empty {description}: {path}")


def require_trec_identifier(value: str, *, path: Path, line_number: int, field: str) -> None:
    if not value:
        raise CorpusBuildError(f"Empty {field} in {path}:{line_number}")


def read_positive_qrels(path: Path) -> set[str]:
    """Return unique docids with relevance strictly greater than zero."""
    require_nonempty_file(path, "qrels file")
    positive_docids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    row_count = 0

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            fields = line.split()
            if len(fields) != 4:
                raise CorpusBuildError(
                    f"Expected 4 qrels fields in {path}:{line_number}, got {len(fields)}"
                )
            query_id, _iteration, docid, relevance_text = fields
            require_trec_identifier(query_id, path=path, line_number=line_number, field="query id")
            require_trec_identifier(docid, path=path, line_number=line_number, field="docid")
            pair = (query_id, docid)
            if pair in seen_pairs:
                raise CorpusBuildError(
                    f"Duplicate qrels entry for query={query_id!r}, docid={docid!r} "
                    f"in {path}:{line_number}"
                )
            seen_pairs.add(pair)
            try:
                relevance = int(relevance_text)
            except ValueError as error:
                raise CorpusBuildError(
                    f"Invalid qrels relevance in {path}:{line_number}: {relevance_text!r}"
                ) from error
            if relevance > 0:
                positive_docids.add(docid)

    if row_count == 0:
        raise CorpusBuildError(f"Qrels file has no records: {path}")
    return positive_docids


def read_top_k_run_docids(path: Path, *, top_k: int) -> set[str]:
    """Strictly parse a TREC top-*k* run and return its unique document IDs."""
    require_nonempty_file(path, "TREC run")
    if top_k <= 0:
        raise CorpusBuildError(f"top_k must be positive, got {top_k}")

    docids: set[str] = set()
    per_query: dict[str, tuple[set[int], set[str]]] = {}
    run_name: str | None = None
    row_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            fields = line.split()
            if len(fields) != 6:
                raise CorpusBuildError(
                    f"Expected 6 TREC run fields in {path}:{line_number}, got {len(fields)}"
                )
            query_id, iteration, docid, rank_text, score_text, current_run_name = fields
            require_trec_identifier(query_id, path=path, line_number=line_number, field="query id")
            require_trec_identifier(docid, path=path, line_number=line_number, field="docid")
            require_trec_identifier(
                current_run_name, path=path, line_number=line_number, field="run name"
            )
            if iteration != "Q0":
                raise CorpusBuildError(
                    f"Expected Q0 in the second TREC field at {path}:{line_number}"
                )
            try:
                rank = int(rank_text)
                score = float(score_text)
            except ValueError as error:
                raise CorpusBuildError(
                    f"Invalid rank or score in {path}:{line_number}"
                ) from error
            if rank <= 0 or rank > top_k or not math.isfinite(score):
                raise CorpusBuildError(
                    f"Invalid top-{top_k} rank or score in {path}:{line_number}"
                )
            if run_name is None:
                run_name = current_run_name
            elif current_run_name != run_name:
                raise CorpusBuildError(f"Multiple run names found in {path}")

            ranks, query_docids = per_query.setdefault(query_id, (set(), set()))
            if rank in ranks or docid in query_docids:
                raise CorpusBuildError(
                    f"Duplicate rank or docid for query={query_id!r} in {path}:{line_number}"
                )
            ranks.add(rank)
            query_docids.add(docid)
            docids.add(docid)

    if row_count == 0:
        raise CorpusBuildError(f"TREC run has no records: {path}")
    for query_id, (ranks, _query_docids) in per_query.items():
        if ranks != set(range(1, len(ranks) + 1)):
            raise CorpusBuildError(
                f"Ranks for query={query_id!r} in {path} are not a contiguous 1..N sequence"
            )
    return docids


def load_selection_inputs(paths: SelectionPaths, *, top_k: int) -> SelectionInputs:
    """Load the six frozen selection sources without consulting any other runs."""
    dev_relevant_docids = read_positive_qrels(paths.dev_qrels)
    test_relevant_docids = read_positive_qrels(paths.test_qrels)
    bm25_dev_docids = read_top_k_run_docids(paths.bm25_dev_run, top_k=top_k)
    bm25_test_docids = read_top_k_run_docids(paths.bm25_test_run, top_k=top_k)
    e5_dev_docids = read_top_k_run_docids(paths.e5_dev_run, top_k=top_k)
    e5_test_docids = read_top_k_run_docids(paths.e5_test_run, top_k=top_k)

    source_docids = {
        "positive_dev_qrels": dev_relevant_docids,
        "positive_test_qrels": test_relevant_docids,
        f"bm25_dev_top_{top_k}": bm25_dev_docids,
        f"bm25_test_top_{top_k}": bm25_test_docids,
        f"e5_dev_top_{top_k}": e5_dev_docids,
        f"e5_test_top_{top_k}": e5_test_docids,
    }
    required_docids = set().union(*source_docids.values())
    return SelectionInputs(
        required_docids=required_docids,
        dev_relevant_docids=dev_relevant_docids,
        test_relevant_docids=test_relevant_docids,
        source_counts={name: len(docids) for name, docids in source_docids.items()},
        source_paths={
            "positive_dev_qrels": str(paths.dev_qrels),
            "positive_test_qrels": str(paths.test_qrels),
            f"bm25_dev_top_{top_k}": str(paths.bm25_dev_run),
            f"bm25_test_top_{top_k}": str(paths.bm25_test_run),
            f"e5_dev_top_{top_k}": str(paths.e5_dev_run),
            f"e5_test_top_{top_k}": str(paths.e5_test_run),
        },
    )


def iter_corpus_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream validated JSON objects from a gzip JSONL corpus."""
    require_nonempty_file(path, "source corpus")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CorpusBuildError(
                        f"Malformed corpus JSON in {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise CorpusBuildError(
                        f"Expected a JSON object in {path}:{line_number}"
                    )
                yield line_number, record
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise CorpusBuildError(f"Corrupt gzip source corpus {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise CorpusBuildError(f"Source corpus is not valid UTF-8 {path}: {error}") from error


def validated_document(record: dict[str, Any], *, path: Path, line_number: int) -> tuple[str, str, str]:
    """Extract the three preserved fields, rejecting incomplete corpus records."""
    try:
        docid = record["docid"]
        title = record["title"]
        text = record["text"]
    except KeyError as error:
        if error.args[0] == "docid":
            raise CorpusBuildError(
                f"Missing or non-string docid in source corpus {path}:{line_number}"
            ) from error
        raise CorpusBuildError(
            f"Missing {error.args[0]!r} in source corpus {path}:{line_number}"
        ) from error
    if not isinstance(docid, str) or not docid:
        raise CorpusBuildError(f"Missing or non-string docid in source corpus {path}:{line_number}")
    if not isinstance(title, str) or not isinstance(text, str):
        raise CorpusBuildError(
            f"Non-string title or text in source corpus {path}:{line_number}"
        )
    return docid, title, text


def file_snapshot(path: Path) -> FileSnapshot:
    stat = path.stat()
    return FileSnapshot(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def require_unchanged_source(path: Path, expected: FileSnapshot) -> None:
    if file_snapshot(path) != expected:
        raise CorpusBuildError(f"Source corpus changed while building: {path}")


def sample_and_validate_source(
    *,
    corpus_path: Path,
    required_docids: set[str],
    random_slots: int,
    random_seed: int,
    duplicate_db_path: Path,
) -> tuple[int, set[str], list[str]]:
    """First pass: detect source duplicates and choose a uniform ID reservoir."""
    random_generator = random.Random(random_seed)
    reservoir: list[str] = []
    required_seen: set[str] = set()
    eligible_count = 0
    source_count = 0

    with sqlite3.connect(duplicate_db_path) as connection:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE seen_docids (docid TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        for line_number, record in iter_corpus_records(corpus_path):
            docid, _title, _text = validated_document(
                record, path=corpus_path, line_number=line_number
            )
            source_count += 1
            try:
                connection.execute("INSERT INTO seen_docids VALUES (?)", (docid,))
            except sqlite3.IntegrityError as error:
                raise CorpusBuildError(
                    f"Duplicate source docid {docid!r} in {corpus_path}:{line_number}"
                ) from error

            if docid in required_docids:
                required_seen.add(docid)
                continue

            eligible_count += 1
            if len(reservoir) < random_slots:
                reservoir.append(docid)
                continue
            replacement_index = random_generator.randrange(eligible_count)
            if replacement_index < random_slots:
                reservoir[replacement_index] = docid

    missing_required_docids = required_docids - required_seen
    if missing_required_docids:
        examples = ", ".join(repr(docid) for docid in sorted(missing_required_docids)[:5])
        raise CorpusBuildError(
            f"{len(missing_required_docids):,} required source docids are missing; examples: {examples}"
        )
    if eligible_count < random_slots:
        raise CorpusBuildError(
            f"Source corpus has only {eligible_count:,} eligible non-required documents, "
            f"but {random_slots:,} are needed"
        )
    return source_count, required_seen, reservoir


def write_deterministic_corpus(
    *, corpus_path: Path, output_path: Path, selected_docids: set[str]
) -> tuple[int, int]:
    """Second pass: emit selected documents in original source order with stable gzip bytes."""
    temporary_path: Path | None = None
    emitted_docids: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".corpus.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=temporary_file, mtime=0
            ) as compressed_stream:
                for line_number, record in iter_corpus_records(corpus_path):
                    docid, title, text = validated_document(
                        record, path=corpus_path, line_number=line_number
                    )
                    if docid not in selected_docids:
                        continue
                    if docid in emitted_docids:
                        raise CorpusBuildError(
                            f"Duplicate output docid {docid!r} at source line {line_number}"
                        )
                    emitted_docids.add(docid)
                    output_record = {"docid": docid, "title": title, "text": text}
                    encoded_record = json.dumps(
                        output_record, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    compressed_stream.write(encoded_record + b"\n")

        if emitted_docids != selected_docids:
            missing_docids = selected_docids - emitted_docids
            examples = ", ".join(repr(docid) for docid in sorted(missing_docids)[:5])
            raise CorpusBuildError(
                f"Selected docids disappeared before output; examples: {examples}"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
        return len(emitted_docids), len(set(emitted_docids))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage(required: set[str], present: set[str]) -> dict[str, int | float]:
    covered = len(required & present)
    total = len(required)
    return {
        "covered": covered,
        "total": total,
        "fraction": covered / total if total else 1.0,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".manifest.", suffix=".tmp", dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_reduced_corpus(
    *,
    corpus_path: Path,
    output_dir: Path,
    selection_paths: SelectionPaths,
    target_size: int = DEFAULT_TARGET_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    expected_source_count: int = DEFAULT_EXPECTED_SOURCE_COUNT,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Build and validate the reduced corpus, returning the written manifest."""
    if target_size <= 0:
        raise CorpusBuildError(f"target_size must be positive, got {target_size}")
    if expected_source_count <= 0:
        raise CorpusBuildError(
            f"expected_source_count must be positive, got {expected_source_count}"
        )

    selection_inputs = load_selection_inputs(selection_paths, top_k=top_k)
    required_count = len(selection_inputs.required_docids)
    if required_count >= target_size:
        raise CorpusBuildError(
            f"Required union has {required_count:,} docids, which must be smaller than "
            f"target_size={target_size:,}"
        )
    random_slots = target_size - required_count

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corpus.jsonl.gz"
    manifest_path = output_dir / "manifest.json"
    existing_outputs = [path for path in (output_path, manifest_path) if path.exists()]
    if existing_outputs:
        names = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite existing reduced-corpus output: {names}")

    source_before = file_snapshot(corpus_path)
    with tempfile.TemporaryDirectory(prefix=".build-reduced-corpus-", dir=output_dir) as temporary_dir:
        source_count, required_seen, reservoir = sample_and_validate_source(
            corpus_path=corpus_path,
            required_docids=selection_inputs.required_docids,
            random_slots=random_slots,
            random_seed=random_seed,
            duplicate_db_path=Path(temporary_dir) / "source-docids.sqlite3",
        )
        if source_count != expected_source_count:
            raise CorpusBuildError(
                f"Expected {expected_source_count:,} source documents, found {source_count:,}"
            )
        require_unchanged_source(corpus_path, source_before)

        selected_docids = selection_inputs.required_docids | set(reservoir)
        if len(selected_docids) != target_size:
            raise CorpusBuildError(
                f"Selection has {len(selected_docids):,} unique docids, expected {target_size:,}"
            )
        output_count, unique_output_count = write_deterministic_corpus(
            corpus_path=corpus_path, output_path=output_path, selected_docids=selected_docids
        )
        require_unchanged_source(corpus_path, source_before)

    if output_count != target_size or unique_output_count != target_size:
        raise CorpusBuildError(
            f"Generated output has {output_count:,} rows and {unique_output_count:,} unique docids; "
            f"expected {target_size:,}"
        )

    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_corpus_path": str(corpus_path),
        "target_size": target_size,
        "random_seed": random_seed,
        "top_k_per_run": top_k,
        "sampling": {
            "algorithm": "Algorithm R reservoir sampling over eligible non-required docids",
            "corpus_passes": 2,
            "output_order": "original source-corpus order",
        },
        "expected_source_corpus_count": expected_source_count,
        "actual_source_corpus_count": source_count,
        "required_union_count": required_count,
        "randomly_selected_count": len(reservoir),
        "final_output_count": output_count,
        "unique_output_docid_count": unique_output_count,
        "selection_sources": {
            name: {"path": selection_inputs.source_paths[name], "unique_docid_count": count}
            for name, count in selection_inputs.source_counts.items()
        },
        "positive_dev_qrels_count": len(selection_inputs.dev_relevant_docids),
        "positive_test_qrels_count": len(selection_inputs.test_relevant_docids),
        "dev_relevant_document_coverage": coverage(
            selection_inputs.dev_relevant_docids, required_seen
        ),
        "test_relevant_document_coverage": coverage(
            selection_inputs.test_relevant_docids, required_seen
        ),
        "total_required_document_coverage": coverage(
            selection_inputs.required_docids, required_seen
        ),
        "corpus_jsonl_gz_sha256": sha256_file(output_path),
    }
    write_manifest(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dev-qrels", type=Path, default=DEFAULT_DEV_QRELS_PATH)
    parser.add_argument("--test-qrels", type=Path, default=DEFAULT_TEST_QRELS_PATH)
    parser.add_argument("--bm25-dev-run", type=Path, default=DEFAULT_BM25_DEV_RUN_PATH)
    parser.add_argument("--bm25-test-run", type=Path, default=DEFAULT_BM25_TEST_RUN_PATH)
    parser.add_argument("--e5-dev-run", type=Path, default=DEFAULT_E5_DEV_RUN_PATH)
    parser.add_argument("--e5-test-run", type=Path, default=DEFAULT_E5_TEST_RUN_PATH)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--expected-source-count", type=int, default=DEFAULT_EXPECTED_SOURCE_COUNT
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_reduced_corpus(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        selection_paths=SelectionPaths(
            dev_qrels=args.dev_qrels,
            test_qrels=args.test_qrels,
            bm25_dev_run=args.bm25_dev_run,
            bm25_test_run=args.bm25_test_run,
            e5_dev_run=args.e5_dev_run,
            e5_test_run=args.e5_test_run,
        ),
        target_size=args.target_size,
        random_seed=args.seed,
        expected_source_count=args.expected_source_count,
        top_k=args.top_k,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
