# Research data and portable manifest

`raw/mr-tydi-russian/` holds the local Russian Mr. TyDi source corpus, topics, and qrels. The historical full-corpus experiment used **9,597,504 documents**. `processed/mr-tydi-russian-reduced-1m/corpus.jsonl.gz` is the frozen **1,000,000-document** collection used for the five-method modern retriever validation. Raw data and the reduced corpus binary are intentionally excluded from Git.

The tracked [reduced-1m-manifest.json](reduced-1m-manifest.json) is a portable copy of the local construction manifest. Machine-specific absolute paths were replaced with repository-relative paths; all counts, seed, selection protocol, and the compressed-corpus SHA-256 were preserved. The local original `processed/mr-tydi-russian-reduced-1m/manifest.json` and frozen corpus remain unchanged.

The required **369,957** documents came from positive DEV and TEST qrels and historical BM25 and E5 top-100 DEV/TEST runs. A seed-42 reservoir sample supplied **630,043** more documents. All 1,332 positive DEV and 1,100 positive TEST qrels documents were retained, and the output contains 1,000,000 unique docids in source-corpus order. The manifest lists the individual source counts and the expected SHA-256. Verify a local binary without modifying it:

```sh
shasum -a 256 data/processed/mr-tydi-russian-reduced-1m/corpus.jsonl.gz
```

The digest should be `4537a2cc012121b92ecd06ed3c248519050a469db82831f9718220ebb84edca3`.

The reduced collection was **TEST-informed**: positive TEST qrels and historical TEST rankings contributed to its construction. It is not a representative random sample of the full corpus. The reduced TEST evaluation used frozen retriever parameters and a common fixed collection, but it is not fully independent of collection construction. Interpret all reduced results and paired statistics conditionally on this selection; they do not establish full-corpus BGE-M3 performance. See the [completed protocol](../docs/modern-retriever-validation-protocol.md) for frozen configurations, DEV/TEST metrics, paired tests, timing, and reproducibility details.
