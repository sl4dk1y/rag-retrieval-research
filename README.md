# RAG Retrieval Research

Репозиторий исследования методов извлечения русскоязычных документов для
Retrieval-Augmented Generation. На текущем этапе реализован только
воспроизводимый **BM25 baseline** для русскоязычной части Mr. TyDi: построение
индекса, поиск и оценка. Dense Retrieval, Hybrid Retrieval и генерация ответов
LLM в этот baseline не входят.

## Требования

- Python 3.11 или новее;
- локальные файлы русскоязычного Mr. TyDi;
- достаточно оперативной памяти и свободного места для полного корпуса.

Зависимости закреплены точными версиями в `requirements.txt`. Русский текст
приводится к нижнему регистру и обрабатывается Snowball stemmer `russian` из
PyStemmer, который напрямую поддерживается токенизатором `bm25s`. Стоп-слова
не удаляются. Используется стандартный для `bm25s` вариант Lucene BM25 с
`k1=1.2` и `b=0.75`.

Создание окружения:

```bash
python3 --version  # должно быть 3.11+
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Ожидается следующая локальная структура:

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

`data/raw/` игнорируется Git. Код читает `corpus.jsonl.gz` построчно и не
создаёт список JSON-словарей всего корпуса. Для индексации формируется ровно
`title + " " + text`; параллельный файл `indexes/bm25/docids.txt` сохраняет
соответствие внутренней позиции `bm25s` исходному `docid` Mr. TyDi.

## Запуск BM25 baseline

Первый рабочий эксперимент выполняется на `dev`:

```bash
python src/build_bm25_index.py
python src/run_bm25.py --split dev --top-k 100
python src/evaluate_run.py --split dev
```

Единая команда делает те же шаги и строит индекс только при его отсутствии:

```bash
python src/run_bm25_experiment.py --split dev --top-k 100
```

Повторный запуск `build_bm25_index.py` использует готовый индекс. Для
осознанного перестроения доступен флаг `--force`. Незавершённый индекс не
используется: отдельный индексатор просит явный `--force`, а единая команда
безопасно заменяет такой каталог только после успешного нового построения.

## Результаты

Для `dev` создаются:

```text
indexes/bm25/                         локальный индекс и mapping docid
results/runs/bm25_dev.trec            top-100 в формате TREC run
results/tables/bm25_dev_metrics.json  Recall@1/5/10, MRR@10, nDCG@10
results/tables/bm25_dev_timing.json   время чтения, токенизации, индекса и поиска
```

Строка TREC run имеет формат:

```text
query_id Q0 doc_id rank score bm25
```

Recall, MRR и nDCG рассчитаны собственной небольшой реализацией: метрики
усредняются по всем запросам topics; документы с relevance > 0 считаются
релевантными, а nDCG сохраняет целочисленные graded relevance из qrels.

`indexes/` уже игнорируется Git. Содержимое `data/raw/` и сам индекс не следует
добавлять в репозиторий.

## Структура кода

```text
src/data_utils.py              чтение и строгая проверка данных/файлов
src/build_bm25_index.py        потоковое чтение корпуса и построение индекса
src/run_bm25.py                поиск и запись TREC run
src/evaluate_run.py            проверка run и расчёт метрик
src/run_bm25_experiment.py     единый end-to-end запуск
src/inspect_dataset.py         исходная утилита первичного осмотра датасета
```
