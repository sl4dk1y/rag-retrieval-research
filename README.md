# RAG Retrieval Research

Репозиторий студенческого исследования методов извлечения русскоязычных документов для Retrieval-Augmented Generation (RAG).

На текущем этапе реализованы и проверены на русскоязычной части Mr. TyDi:

- **BM25** — классический разреженный лексический поиск;
- **Dense Retrieval** — семантический поиск на `intfloat/multilingual-e5-small` + FAISS;
- единая оценка retrieval-методов по Recall, MRR и nDCG;
- эксперименты с параметром `nprobe` для Dense Retrieval.

Следующий этап — **Hybrid Retrieval** с объединением BM25 и Dense Retrieval через Reciprocal Rank Fusion (RRF).

## Требования

- Python 3.11+;
- локальные файлы русскоязычной части Mr. TyDi;
- достаточно RAM и свободного места для индексов;
- на Apple Silicon Dense Retrieval использует PyTorch MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

```text
data/raw/mr-tydi-russian/
├── corpus.jsonl.gz
├── train.jsonl.gz
├── dev.jsonl.gz
├── test.jsonl.gz
└── ir-format-data/
    ├── topics.train.txt
    ├── topics.dev.txt
    ├── topics.test.txt
    ├── qrels.train.txt
    ├── qrels.dev.txt
    └── qrels.test.txt
```

`data/raw/` игнорируется Git. Полный русскоязычный корпус содержит **9 597 504 документа**.

## BM25 baseline

Конфигурация:

- `bm25s`;
- Lucene BM25;
- `k1 = 1.2`;
- `b = 0.75`;
- русский Snowball stemmer;
- стоп-слова не удаляются;
- индексируется `title + " " + text`.

Запуск:

```bash
python src/build_bm25_index.py
python src/run_bm25.py --split dev --top-k 100
```

Оценка:

```bash
python src/evaluate_run.py   --split dev   --run results/runs/bm25_dev.trec   --docids indexes/bm25/docids.txt   --metrics-output results/tables/bm25_dev_metrics.json
```

Результаты BM25 на `dev`:

| Метрика | Значение |
| --- | ---: |
| Recall@1 | 0.110545 |
| Recall@5 | 0.224727 |
| Recall@10 | 0.277818 |
| MRR@10 | 0.158763 |
| nDCG@10 | 0.186966 |

## Dense Retrieval

Основная модель:

```text
intfloat/multilingual-e5-small
```

Конфигурация:

- embedding dimension: **384**;
- документы: `passage: <title> <text>`;
- запросы: `query: <text>`;
- embeddings нормализуются;
- FAISS `IVFScalarQuantizer`;
- SQ8;
- inner product;
- `nlist = 2048`;
- обучение FAISS на 100 000 документах;
- batch size: **8**;
- полный индекс: около **3.5 GB**;
- mapping `docids.txt`: около **91 MB**;
- peak RSS при построении: около **3.74 GiB**;
- скорость кодирования полного корпуса: **92.64 docs/s**.

Полный индекс хранится локально в:

```text
indexes/dense-e5-small/
```

Построение:

```bash
python src/build_dense_index.py   --train-docs 100000   --nlist 2048   --batch-size 8
```

Поиск:

```bash
python src/run_dense.py   --split dev   --top-k 100   --nprobe 256
```

## Dense nprobe experiments

| nprobe | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.380364 | 0.670545 | 0.744000 | 0.503056 | 0.561469 |
| 128 | 0.388364 | 0.688000 | 0.771636 | 0.516823 | 0.578498 |
| 256 | 0.394909 | 0.701818 | 0.784727 | 0.525663 | 0.588376 |
| 512 | 0.401455 | 0.713455 | 0.796364 | 0.533988 | 0.597516 |

`nprobe=512` даёт максимальное качество из протестированных конфигураций. Для дальнейших экспериментов **`nprobe=256` используется как основной компромисс между качеством и вычислительной стоимостью**.

## Текущее сравнение

| Метод | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.110545 | 0.224727 | 0.277818 | 0.158763 | 0.186966 |
| Dense E5, nprobe=256 | 0.394909 | 0.701818 | 0.784727 | 0.525663 | 0.588376 |
| Dense E5, nprobe=512 | 0.401455 | 0.713455 | 0.796364 | 0.533988 | 0.597516 |

Dense Retrieval существенно превосходит текущий BM25 baseline по всем основным метрикам на `dev`.

## Hybrid Retrieval — следующий этап

План:

```text
BM25 ranking
      +
Dense E5 ranking
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
Hybrid ranking
```

Hybrid Retrieval не требует повторного построения индексов. Он будет объединять уже сохранённые TREC run-файлы BM25 и Dense.

## Метрики

Во всех основных экспериментах используются:

- Recall@1;
- Recall@5;
- Recall@10;
- MRR@10;
- nDCG@10;
- время поиска.

`src/evaluate_run.py` является общим evaluator и не привязан к конкретному retrieval-методу.

## Структура кода

```text
src/
├── inspect_dataset.py
├── data_utils.py
├── build_bm25_index.py
├── run_bm25.py
├── run_bm25_experiment.py
├── benchmark_dense.py
├── dense_utils.py
├── build_dense_index.py
├── run_dense.py
└── evaluate_run.py
```

Текущий активный этап: **Hybrid Retrieval (RRF)**.
