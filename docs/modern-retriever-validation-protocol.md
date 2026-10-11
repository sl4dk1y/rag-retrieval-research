# Modern Retriever Validation — completed protocol

## Scope and collection

The historical Russian Mr. TyDi experiment used **9,597,504 documents**. This completed extension evaluated five methods on a separate, fixed **1,000,000-document** collection. Historical full-corpus results remain unchanged and are reported separately in [README](../README.md) and [final_results.json](../results/final/final_results.json). Scores across the two collections are not a model-only comparison.

The reduced collection was constructed from positive **DEV and TEST qrels**, historical **BM25 DEV and TEST top-100** runs, historical **E5 DEV and TEST top-100** runs, and seeded random sampling (Algorithm R, seed **42**) from eligible remaining documents. The required union had **369,957** documents; **630,043** were randomly selected. Output retained original source-corpus order. All 1,332 positive DEV and 1,100 positive TEST qrels documents are included. The selection counts, input paths, and compressed-corpus SHA-256 are preserved in the portable [manifest](../data/reduced-1m-manifest.json).

**Methodological limitation:** TEST positives and historical TEST retrieval results informed collection selection. BGE-M3 retrieval parameters were frozen before TEST evaluation, and all five methods were evaluated on the same fixed collection; nonetheless, reduced TEST is not fully independent of collection construction. This collection is not a random representative sample of the original 9.6M documents. Statistical results are conditional on this selected collection and do not establish full-corpus BGE-M3 performance.

## Frozen methods

Every method returned top-100 rankings. BM25 used `bm25s` Lucene BM25, `k1=1.2`, `b=0.75`, Russian Snowball stemming, and `title + " " + text`. Both dense indexes used normalized vectors, FAISS `IVFScalarQuantizer` with SQ8 and inner product, `nlist=2048`, 100,000 training documents, and search `nprobe=256`.

| Method | Frozen representation and settings |
| --- | --- |
| BM25 | Lexical settings above |
| E5 | `intfloat/multilingual-e5-small`, 384 dimensions, `passage: ` documents, `query: ` queries, MPS, index batch 8 |
| E5 + BM25 | Weighted RRF of their top-100 runs: `k=60`, dense weight 1.0, BM25 weight 0.025 |
| BGE-M3 | `BAAI/bge-m3` dense mode, 1,024 dimensions, `title + " " + text` documents, raw queries, MPS, index batch 2, query batch 16, maximum sequence length 512 |
| BGE-M3 + BM25 | The same frozen weighted RRF of BGE-M3 and BM25 top-100 runs |

The actual historical full-corpus E5 index used `nlist=2048`. The historical builder retains a **default** of `nlist=4096`; that default is not the configuration of the evaluated index. No historical builder default or index artifact was changed.

The local environment was a MacBook Air with Apple M4, 16 GB unified memory, macOS, Python 3.12, and PyTorch MPS. Package versions are in [requirements.txt](../requirements.txt). Model names were recorded, but exact model revisions and model-cache contents were not pinned, limiting bitwise reproduction elsewhere.

## Observed reduced-collection quality

The DEV split contained 1,375 queries. Values below are from the saved aggregate metric JSON files in [results/reduced-1m/tables](../results/reduced-1m/tables/).

| Reduced DEV method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.117091 | 0.236364 | 0.287273 | 0.165755 | 0.194526 |
| E5 | 0.395636 | 0.705455 | 0.786182 | 0.528073 | 0.590686 |
| E5 + BM25 | 0.393455 | 0.705455 | 0.786182 | 0.525873 | 0.589017 |
| BGE-M3 | 0.474909 | 0.773818 | 0.857455 | 0.605035 | 0.666260 |
| BGE-M3 + BM25 | 0.467636 | 0.771636 | 0.858182 | 0.600214 | 0.662785 |

The TEST split contained 995 queries. This five-method table reflects the preserved TEST metric JSON; neither metrics nor runs were regenerated.

| Reduced TEST method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.173199 | 0.303183 | 0.369347 | 0.247903 | 0.268482 |
| E5 | 0.437521 | 0.740536 | 0.813065 | 0.599389 | 0.641458 |
| E5 + BM25 | 0.432663 | 0.738023 | 0.812563 | 0.598179 | 0.640806 |
| BGE-M3 | 0.479899 | 0.800335 | 0.861307 | 0.651545 | 0.692026 |
| BGE-M3 + BM25 | 0.469849 | 0.797822 | 0.861307 | 0.646244 | 0.688075 |

The historical full-corpus TEST scores for BM25, E5, and E5 + BM25 are not interchangeable with these reduced-collection scores.

## Paired TEST analysis

Three comparisons were predefined: **BGE-M3 vs E5**, **E5 + BM25 vs E5**, and **BGE-M3 + BM25 vs BGE-M3**. For each of five metrics, the analysis used query-level A − B differences for the same 995 queries, **10,000 paired bootstrap samples** for **95% percentile confidence intervals**, and **10,000 paired sign-flip randomizations** for **two-sided p-values**, with seed **42**. The unchanged full output is [paired_test_statistics.json](../results/reduced-1m/tables/paired_test_statistics.json).

| A − B | Metric | Difference | 95% CI | Nominal p |
| --- | --- | ---: | ---: | ---: |
| BGE-M3 − E5 | Recall@1 | +0.042379 | [0.015406, 0.070017] | 0.003400 |
| BGE-M3 − E5 | Recall@5 | +0.059799 | [0.039698, 0.080402] | 0.000100 |
| BGE-M3 − E5 | Recall@10 | +0.048241 | [0.030318, 0.067002] | 0.000100 |
| BGE-M3 − E5 | MRR@10 | +0.052156 | [0.032402, 0.072332] | 0.000100 |
| BGE-M3 − E5 | nDCG@10 | +0.050568 | [0.033900, 0.067630] | 0.000100 |
| E5 + BM25 − E5 | Recall@1 | −0.004858 | [−0.017420, 0.007538] | 0.484652 |
| E5 + BM25 − E5 | Recall@5 | −0.002513 | [−0.006533, 0.001005] | 0.310269 |
| E5 + BM25 − E5 | Recall@10 | −0.000503 | [−0.003015, 0.001508] | 1.000000 |
| E5 + BM25 − E5 | MRR@10 | −0.001210 | [−0.008199, 0.005783] | 0.744326 |
| E5 + BM25 − E5 | nDCG@10 | −0.000652 | [−0.005575, 0.004270] | 0.801820 |
| BGE-M3 + BM25 − BGE-M3 | Recall@1 | −0.010050 | [−0.023786, 0.003350] | 0.157084 |
| BGE-M3 + BM25 − BGE-M3 | Recall@5 | −0.002513 | [−0.007538, 0.002010] | 0.416758 |
| BGE-M3 + BM25 − BGE-M3 | Recall@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 |
| BGE-M3 + BM25 − BGE-M3 | MRR@10 | −0.005302 | [−0.012689, 0.001899] | 0.153385 |
| BGE-M3 + BM25 − BGE-M3 | nDCG@10 | −0.003950 | [−0.009133, 0.001148] | 0.135086 |

All **15 p-values are nominal and unadjusted for multiple comparisons**. On this selected collection, BGE-M3 outperformed E5 on all five TEST metrics, and the paired results support that advantage at the nominal significance level. This supports H1 for this fixed collection and these configurations. No statistically significant improvement from fixed weighted RRF was detected for either dense retriever, so H2 is not supported. This does not prove the null hypothesis, equivalence, or significant degradation; it also does not imply that hybrid retrieval is ineffective in general. Holm's step-down correction could be applied to the five recorded primary BGE-M3 vs E5 p-values if familywise control is required; no new exploratory tests are reported.

## Measured computational cost

The BGE-M3 index metadata records **80,853.38 seconds** of main-corpus model encoding (about **22 h 28 min**) and **9,932.68 seconds** of training-sample model encoding (about **2 h 46 min**). These are separate encoding measurements. **Total build duration was 91,196.10 seconds** (about **25 h 20 min**), including other build steps; it is not document encoding time. The exact measured JSON is preserved in [dense_bge_m3_build_timing.json](../results/reduced-1m/tables/dense_bge_m3_build_timing.json). The FAISS file size was 1,040,413,368 bytes and recorded peak process RSS was 1,032,224,768 bytes. Per-split retrieval timing JSON is retained in the same tables directory. These measurements describe the local Apple M4/MPS environment. Earlier 9.6M-corpus BGE-M3 feasibility extrapolations were **estimates**, not measured full-corpus build durations.

The [full-corpus token-length analysis](../results/tables/bge_m3_token_lengths_full.json) counted 36,714 of 9,597,504 source documents above 512 tokens under the BGE-M3 tokenizer. The frozen maximum sequence length can therefore truncate some inputs. The BGE-M3 benchmark JSON files in `results/tables/` describe feasibility, not retrieval quality.

## BGE-M3 docid-mapping incident and integrity

The BGE-M3 FAISS index was successfully built, but `docids.txt` was unexpectedly empty. The exact cause remains unknown. The mapping was recovered by streaming the frozen reduced corpus. The compressed corpus SHA-256 was verified as `4537a2cc012121b92ecd06ed3c248519050a469db82831f9718220ebb84edca3`. The recovered mapping contained exactly **1,000,000 unique docids**, in the same order as the frozen corpus, and the index contained **1,000,000 vectors**. A read-only final check reconfirmed mapping order and uniqueness. The verified mapping SHA-256 is `3fff4ca63a3705ff5b518711f4827634f0f936e811205f8d98f8e1deb8e0b90d`. The mapping and index remain local, excluded from Git. The subsequent builder and runner integrity checks require complete artifacts and matching mapping, metadata, and index counts before use.

## Reproducibility and artifact boundaries

Use Python 3.12 and install [requirements.txt](../requirements.txt). Obtain Russian Mr. TyDi raw corpus, topics, and qrels under `data/raw/mr-tydi-russian/`; raw data, model weights, and indexes are deliberately absent from Git. The historical top-100 selection runs are tracked under `results/runs/`. Verify a local reduced corpus and recovered mapping against the manifest and checksum above:

```sh
shasum -a 256 data/processed/mr-tydi-russian-reduced-1m/corpus.jsonl.gz
shasum -a 256 indexes/mr-tydi-russian-reduced-1m/dense-bge-m3/docids.txt
```

For a fresh independent reproduction, `src/build_reduced_corpus.py` defines the seeded selection, `src/build_bm25_index.py`, `src/build_dense_index.py`, and `src/build_bge_m3_index.py` define index builds, and `src/run_bm25.py`, `src/run_dense.py`, `src/run_bge_m3.py`, and `src/run_hybrid.py` define retrieval. Use separate output directories and the frozen settings above. Saved top-100 DEV and TEST runs, aggregate metrics, retrieval timing, and paired statistics are already under `results/reduced-1m/`, so inspecting the completed experiment needs no new retrieval. To independently recalculate metrics, use `src/evaluate_run.py --split {dev,test} --run <run> --docids <matching-index-docids> --metrics-output <new-path>`. To recalculate the three paired comparisons from saved runs and raw qrels, use `src/analyze_paired_test.py --output <new-path>`. Always choose new output paths: script defaults can point at archived results.

This finalization does not rebuild indexes, rerun retrieval, regenerate metrics, or tune parameters on TEST.
