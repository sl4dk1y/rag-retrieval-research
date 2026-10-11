# План исследования

## Рабочее название статьи

**Сравнительный анализ методов извлечения русскоязычных документов для систем Retrieval-Augmented Generation**

## Цель

Экспериментально сравнить качество и производительность BM25, Dense Retrieval и Hybrid Retrieval на русскоязычной части Mr. TyDi и определить, даёт ли объединение лексического и семантического retrieval устойчивое преимущество.

## Исходный исторический исследовательский вопрос

Какой из рассматриваемых методов обеспечивает наиболее высокое качество извлечения релевантных русскоязычных документов при одинаковых условиях эксперимента?

Исходный вопрос о том, даёт ли weighted hybrid retrieval преимущество над отдельными retriever-ами, сохраняется в этом плане. Завершённая работа включает также сравнение современного dense retriever-а с историческим baseline.

## Гипотезы завершённого исследования

- **H1:** BGE-M3 обеспечивает более высокую retrieval effectiveness, чем `multilingual-e5-small`, на фиксированной reduced-1M коллекции.
- **H2:** добавление лексического BM25 через фиксированный weighted Reciprocal Rank Fusion даёт последовательное улучшение относительно соответствующего standalone dense retriever-а.

На выбранной reduced-1M коллекции наблюдаемые TEST-результаты и номинальные парные тесты поддерживают H1. H2 не поддерживается: статистически значимого улучшения не обнаружено для E5 + BM25 или BGE-M3 + BM25. Отсутствие значимого улучшения не доказывает нулевой эффект, эквивалентность или значимое ухудшение. Эти выводы относятся к проверенным коллекциям и конфигурациям, а не ко всем датасетам, весам fusion или архитектурам.

## Historical full-corpus dataset

Русскоязычная часть **Mr. TyDi**:

- корпус: **9 597 504 документа**;
- `train`;
- `dev`;
- `test`;
- финальный `test`: **995 запросов**.

## Historical full-corpus experimental protocol

1. Все методы оцениваются на одном корпусе и совместимых split.
2. `dev` используется для ablation и выбора параметров.
3. После выбора параметры фиксируются.
4. `test` использовался для финальной оценки на исходном полном корпусе.
5. Параметры retrieval не настраиваются по результатам `test`.
6. Качественный error analysis выполняется после фиксации финальных конфигураций.
7. Финальные результаты агрегируются в repository-tracked JSON/CSV и publication-ready figures.

## Experimental environment

Локальное окружение:

- MacBook Air;
- Apple M4;
- 16 GB unified memory;
- macOS;
- Python 3.12;
- PyTorch MPS для Dense Retrieval.

Аппаратное окружение фиксируется для корректной интерпретации latency и throughput; абсолютные значения производительности не предполагаются универсальными для другого оборудования.

## Methods and frozen configurations

### BM25

- `bm25s`;
- Lucene BM25;
- `k1 = 1.2`;
- `b = 0.75`;
- русский Snowball stemmer;
- top-100.

### Dense Retrieval

- `intfloat/multilingual-e5-small`;
- dimension: **384**;
- normalized embeddings;
- FAISS `IVFScalarQuantizer`;
- SQ8;
- inner product;
- `nlist = 2048`;
- исторический default builder-а `nlist = 4096` отличается от фактически построенного и оценённого индекса `nlist = 2048`;
- `nprobe = 256`;
- top-100.

### Hybrid Retrieval

- weighted Reciprocal Rank Fusion;
- `k = 60`;
- Dense `nprobe = 256`;
- `w_Dense = 1.0`;
- `w_BM25 = 0.025`;
- top-100.

## Historical full-corpus DEV results

| Метод | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.110545 | 0.224727 | 0.277818 | 0.158763 | 0.186966 |
| Dense, nprobe=256 | 0.394909 | 0.701818 | 0.784727 | 0.525663 | 0.588376 |
| Hybrid, w_BM25=0.025 | 0.392727 | 0.704000 | 0.787636 | 0.523740 | 0.587544 |

## Historical full-corpus final TEST results

| Метод | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.164154 | 0.296650 | 0.364824 | 0.239390 | 0.260872 |
| Dense, nprobe=256 | 0.434003 | **0.741039** | **0.810553** | 0.596679 | 0.639182 |
| Hybrid, w_BM25=0.025 | **0.434171** | 0.736013 | 0.809548 | **0.597441** | **0.639626** |

## Error analysis

На `test`, cutoff `@10`:

- `all_success`: **373**;
- `all_fail`: **146**;
- `bm25_success_dense_fail`: **20**;
- `dense_success_bm25_fail`: **455**;
- `dense_success_hybrid_fail`: **1**.

Dense vs Hybrid:

- Hybrid лучше: **58**;
- Dense лучше: **48**;
- одинаковый результат: **889**.

Качественный анализ показывает, что BM25 действительно содержит дополнительный лексический сигнал для отдельных запросов, однако выбранный weighted RRF не обеспечивает систематического улучшения относительно Dense.

## Итоговый вывод

Dense Retrieval существенно превосходит BM25.

Hybrid Retrieval после настройки небольшого веса BM25 достигает практически того же качества, что и Dense. На финальном `test` Hybrid немного лучше по Recall@1, MRR@10 и nDCG@10, тогда как Dense немного лучше по Recall@5 и Recall@10.

Следовательно, исходная гипотеза о безусловном превосходстве Hybrid Retrieval не подтверждается. Результаты указывают скорее на ограниченную комплементарность лексического и семантического сигналов и необходимость более совершенных методов fusion/reranking для реализации потенциального преимущества.

## Статус выполнения

1. Проверка и подготовка Mr. TyDi — **завершено**.
2. BM25 baseline — **завершено**.
3. Dense Retrieval — **завершено**.
4. Dense `nprobe` ablation — **завершено**.
5. Hybrid weighted RRF — **завершено**.
6. Hybrid weight ablation — **завершено**.
7. Сравнение методов на `dev` — **завершено**.
8. Фиксация финальных конфигураций — **завершено**.
9. Независимая оценка на `test` — **завершено**.
10. Качественный error analysis — **завершено**.
11. Агрегирование финальных результатов — **завершено**.
12. Итоговые таблицы — **завершено**.
13. Publication-ready графики — **завершено**.
14. Финализация исследовательской документации — **завершено**.
15. Написание статьи — **следующий основной этап**.

## Final research artifacts

Итоговые данные:

```text
results/final/final_results.json
results/final/test_comparison.csv
results/final/dev_comparison.csv
results/final/dense_nprobe_ablation.csv
results/final/hybrid_weight_ablation.csv
results/final/latency_summary.csv
results/final/error_analysis_summary.csv
```

Графики:

```text
results/final/figures/
```

Качественный анализ:

```text
docs/error-analysis.md
results/analysis/error_analysis_test.csv
results/analysis/error_analysis_test_summary.json
```

## Дополнительный завершённый эксперимент: reduced-1M

Сокращённая фиксированная коллекция содержит **1 000 000** документов из исторических **9 597 504**. На ней оценены пять методов: BM25, E5, E5 + BM25, BGE-M3 dense и BGE-M3 + BM25. Их замороженные model/index/retrieval параметры, результаты DEV и TEST и измеренные вычислительные затраты описаны в [завершённом протоколе](modern-retriever-validation-protocol.md). TEST-таблица из пяти методов также приведена в [README](../README.md); исходные TREC runs и metric/timing JSON хранятся в `results/reduced-1m/`.

На выбранной reduced-1M коллекции BGE-M3 превзошёл E5 по всем пяти TEST-метрикам; парные результаты поддерживают это преимущество на номинальном уровне значимости. Все **15 p-values номинальные, без поправки на множественные сравнения**. Статистически значимого улучшения от фиксированного weighted RRF не обнаружено для обоих dense retriever-ов. Эти результаты не доказывают эквивалентность и не показывают статистически значимого ухудшения. Точные результаты трёх предопределённых парных сравнений — BGE-M3 vs E5, E5 + BM25 vs E5, BGE-M3 + BM25 vs BGE-M3 — приведены в [paired_test_statistics.json](../results/reduced-1m/tables/paired_test_statistics.json); анализ использовал 10 000 paired bootstrap samples для 95% percentile CI и 10 000 двухсторонних paired sign-flip randomizations с seed 42.

Коллекция построена с использованием положительных DEV и TEST qrels, исторических BM25 DEV/TEST top-100 rankings, исторических E5 DEV/TEST top-100 rankings и seeded random sampling. Поэтому reduced TEST не полностью независим от построения коллекции. Она не является репрезентативной случайной выборкой исходных 9 597 504 документов; выводы условны для фиксированной коллекции и не устанавливают full-corpus качество BGE-M3. Для проверки доступны [переносимый manifest](../data/reduced-1m-manifest.json) и [измеренная BGE-M3 build timing metadata](../results/reduced-1m/tables/dense_bge_m3_build_timing.json). Main corpus encoding занял **80 853.38 с** (около **22 ч 28 мин**), training-sample encoding — **9 932.68 с** (около **2 ч 46 мин**), а total build duration составил **91 196.10 с** (около **25 ч 20 мин**) на Apple M4/MPS. Это отдельные показатели; полная продолжительность сборки не является временем кодирования документов.

## Следующий этап

Экспериментальный pipeline считается завершённым и замороженным.

Следующий основной этап — **подготовка текста исследовательской статьи**: формулировка abstract, introduction, related work, methodology, experimental setup, results, discussion, limitations и conclusion на основе уже зафиксированных результатов.

Новые эксперименты не должны добавляться в основной результат без отдельного обоснования, поскольку финальная `test`-оценка уже проведена.
