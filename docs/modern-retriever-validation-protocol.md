# Modern Retriever Validation — Experimental Protocol

## Назначение

Этот документ фиксирует протокол дополнительной проверки современного retriever-а в проекте `rag-retrieval-research`.

Исходный эксперимент уже завершён и не должен переписываться:

- BM25;
- Dense Retrieval на `intfloat/multilingual-e5-small`;
- Hybrid BM25 + Dense через weighted RRF;
- dev ablations;
- frozen test evaluation;
- query-level error analysis;
- финальные таблицы и графики.

Новый этап нужен только для проверки того, изменятся ли выводы исследования при добавлении более современного multilingual retriever-а.

## Research question

Даёт ли более современный multilingual retriever заметный прирост качества относительно замороженного baseline `multilingual-e5-small`, и меняется ли вывод о том, что простое lexical-semantic fusion не даёт устойчивого преимущества над сильным Dense Retrieval?

## Primary candidate

Основной кандидат:

`BAAI/bge-m3`

Первый проход:

**dense retrieval only**

Native sparse и multi-vector режимы BGE-M3 пока не включаются. Их можно исследовать отдельно только после завершения и фиксации dense baseline.

## Frozen historical baselines

### BM25

- `bm25s`
- Lucene BM25
- `k1 = 1.2`
- `b = 0.75`
- Russian Snowball stemmer
- `top_k = 100`

### Dense E5

- `intfloat/multilingual-e5-small`
- dimension: 384
- normalized embeddings
- FAISS IVFScalarQuantizer
- SQ8
- inner product
- `nlist = 2048`
- frozen `nprobe = 256`
- `top_k = 100`

### Hybrid E5 + BM25

- weighted RRF
- `rrf_k = 60`
- `w_dense = 1.0`
- `w_bm25 = 0.025`
- Dense `nprobe = 256`
- `top_k = 100`

## Frozen original test results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.164154 | 0.296650 | 0.364824 | 0.239390 | 0.260872 |
| Dense E5 (`nprobe=256`) | 0.434003 | 0.741039 | 0.810553 | 0.596679 | 0.639182 |
| Hybrid RRF (`w_BM25=0.025`) | 0.434171 | 0.736013 | 0.809548 | 0.597441 | 0.639626 |

Эти результаты не изменяются новым экспериментом.

## Dataset

Mr. TyDi Russian:

- corpus: **9,597,504 documents**
- splits: `train`, `dev`, `test`
- frozen final test: **995 queries**

## Dev / test policy

### Разрешено на dev

- feasibility testing;
- debugging;
- выбор batch size для безопасной работы по памяти;
- выбор engineering-параметров FAISS;
- подбор search-параметров;
- ablation, необходимый для фиксации одной финальной конфигурации.

### Запрещено использовать test для

- выбора варианта модели;
- настройки search parameters;
- выбора fusion weights;
- выбора конфигурации после просмотра test metrics.

### Test rule

После фиксации BGE-M3 configuration на dev выполняется один канонический финальный test run.

Повтор test допустим только при подтверждённой технической ошибке, которая должна быть задокументирована.

## Stage A — feasibility benchmark

Полный корпус из 9.6M документов сразу не кодировать.

Первая задача — проверить, насколько BGE-M3 практически применим на:

- MacBook Air;
- Apple M4;
- 16 GB unified memory;
- macOS;
- MPS при поддержке.

### Проверяем

- загрузку модели;
- MPS compatibility;
- embedding dimension;
- стабильный batch size;
- encoding throughput;
- peak RSS;
- примерное время полного corpus encoding;
- примерный размер embeddings/index.

### Sample sizes

Рекомендуемый порядок:

1. 10,000 документов — smoke test;
2. 100,000 документов — benchmark throughput/memory, если smoke test успешен.

На этих samples нельзя делать выводы о retrieval quality.

### Stage A result

Feasibility benchmark выполнен на 10,000 документах Mr. TyDi Russian.

Проверенная конфигурация:

- model: `BAAI/bge-m3`;
- dense retrieval mode;
- device: Apple M4 / MPS;
- embedding dimension: 1024;
- document embeddings normalized during benchmark encoding;
- batch size: 2;
- timing boundaries синхронизированы через `torch.mps.synchronize()`;
- throughput для оценки полного корпуса рассчитывается по end-to-end corpus-loop time.

После исправления MPS timing выполнен отдельный `max_seq_length` benchmark:

| `max_seq_length` | End-to-end docs/s | Estimated full corpus | Peak RSS |
| ---: | ---: | ---: | ---: |
| 512 | 12.98 | 205.36 h | 1.80 GiB |
| 1024 | 11.88 | 224.50 h | 1.58 GiB |
| 8192 (model default) | 11.85 | 224.90 h | 0.88 GiB |

Для `max_seq_length=512` и model-default `8192` дополнительно выполнялись повторные 10k runs. Наблюдаемое преимущество `512` по throughput повторилось, однако committed benchmark artifacts содержат последние runs каждой конфигурации; абсолютная скорость и peak RSS между отдельными запусками варьировались.

На текущем hardware полный corpus encoding остаётся вычислительно дорогим: даже для `max_seq_length=512` экстраполяция составляет около 205 часов (8.6 суток).

Поэтому Stage A подтверждает техническую совместимость BGE-M3 с MPS и достаточный запас памяти, но вычислительное время остаётся основным ограничением.

`max_seq_length=512` переносится в Stage B как основной efficiency candidate, однако **не считается frozen configuration**. Сокращение sequence length потенциально влияет на retrieval quality, поэтому окончательный выбор должен быть сделан только по `dev` ablation без использования `test`.

`max_seq_length=1024` не показал практически значимого throughput advantage относительно model-default `8192`, поэтому отдельное дальнейшее quality-тестирование этого варианта не является приоритетным.

## Stage B — BGE-M3 dense on dev

Если feasibility подтверждён:

- строится dense BGE-M3 pipeline;
- quality оценивается только на `dev`;
- сравнение проводится с BM25, frozen Dense E5 и frozen E5+BM25 Hybrid.

Метрики:

- Recall@1
- Recall@5
- Recall@10
- MRR@10
- nDCG@10

Engineering metrics:

- encoding time;
- search time;
- average search latency;
- index size;
- peak RSS.

## Search/index policy

По возможности сохранять сопоставимость с исходным Dense pipeline:

- normalized vectors, если это соответствует model usage;
- cosine-equivalent / inner-product search для normalized embeddings;
- FAISS configuration выбирается до test;
- exact configuration сохраняется в metadata.

Если 1024-dimensional BGE-M3 требует другой index configuration из-за памяти или скорости, это должно быть явно задокументировано.

## Stage C — freeze configuration

Перед test фиксируются:

- exact model name;
- model revision, если будет pinning;
- embedding dimension;
- document/query formatting;
- normalization;
- device;
- batch size;
- FAISS index type;
- quantization;
- `nlist`;
- `nprobe`;
- training sample size для FAISS;
- `top_k`;
- прочие model-specific settings.

После этого параметры не меняются по результатам test.

## Stage D — independent test evaluation

После freeze:

- один BGE-M3 dense test run;
- сравнение с historical frozen BM25 / E5 / Hybrid;
- baseline files не перезаписываются;
- новые artifacts получают явные BGE-M3 filenames.

## Stage E — statistical comparison

После test выполнить paired query-level analysis:

1. Dense E5 vs BGE-M3 dense
2. Frozen E5+BM25 Hybrid vs BGE-M3 dense

Желательные результаты:

- per-query reciprocal-rank contributions;
- per-query nDCG@10 contributions;
- paired bootstrap confidence intervals и/или paired randomization test;
- effect-size reporting при необходимости.

Цель — отличить численную разницу от действительно убедительной разницы.

## Optional Stage F — native BGE-M3 hybrid

Этот этап не запускать автоматически.

Только после freeze dense BGE-M3 можно отдельно исследовать:

- BGE-M3 dense;
- BGE-M3 sparse;
- BGE-M3 dense+sparse fusion;
- frozen E5 dense;
- frozen E5+BM25 RRF.

Исследовательский вопрос:

Является ли отсутствие устойчивого выигрыша Hybrid особенностью BM25 + E5 + static RRF, или аналогичный эффект сохраняется при современном unified dense+sparse retriever?

## Success criteria

Новый retriever не обязан победить.

Научно полезны все исходы:

### A. BGE-M3 существенно лучше E5
Современный multilingual retriever меняет уровень baseline.

### B. BGE-M3 лишь немного лучше
Исходные выводы в целом устойчивы к усилению dense retriever.

### C. BGE-M3 сопоставим или хуже
Компактный E5-small остаётся конкурентоспособным на этой постановке.

Результат нельзя подгонять или скрывать.

## Reproducibility requirements

Каждый новый run должен сохранять:

- model identifier;
- software versions;
- device;
- corpus size;
- query count;
- batch size;
- embedding dimension;
- normalization;
- index type;
- FAISS parameters;
- top-k;
- timing;
- peak RSS, если доступен;
- split;
- experiment stage;
- repository-relative output paths.

Не сохранять `/Users/...` в tracked artifacts.

## Git policy

Рабочая ветка:

`experiment/modern-retriever-validation`

Не выполнять автоматически:

- commit;
- push;
- merge;
- tag;
- release;
- deploy;
- history rewrite.

Не делать unrelated refactoring и не изменять historical baseline results.

## Immediate next step

Создать небольшой BGE-M3 feasibility benchmark script, который:

1. загружает `BAAI/bge-m3`;
2. использует MPS при поддержке;
3. кодирует небольшой sample русскоязычного корпуса;
4. измеряет dimension, throughput, elapsed time и peak RSS;
5. не строит полный 9.6M-document index;
6. не использует test metrics.

Только после анализа feasibility решать вопрос о полном BGE-M3 indexing.
