# Semi-Dynamic WiDiD

This is a practical extension of WiDiD from *What is Done is Done: an Incremental Approach to Semantic Shift Detection*.

Original WiDiD keeps historical clusters fixed: past embeddings are packed into cluster centroids, new embeddings are added incrementally, and the resulting clusters are used to compute semantic shift. This version keeps that fast behavior between updates, but after a configurable threshold it performs a historical cluster refresh.

## What changes

The semi-dynamic loop is:

1. Extract contextual embeddings for a target word in the current time period.
2. Assign new embeddings incrementally to the nearest existing sense cluster, or create a new sense cluster.
3. Update cluster centroids, sizes, and aging metadata.
4. When `historical_update_threshold` update batches have arrived, re-cluster retained history.
5. Match refreshed clusters back to old cluster IDs where possible, so sense labels remain stable.
6. Continue incrementally from the refreshed memory.

This relaxes WiDiD's "what is done is done" rule only at controlled intervals.

## Files

- `semi_dynamic_widid.py`: dependency-light implementation using NumPy.
- `semeval_widid_experiment.py`: reads SemEval-style corpora and compares original WiDiD with semi-dynamic WiDiD.
- `README_semi_dynamic_widid.md`: this guide.

## Minimal usage

```python
from semi_dynamic_widid import SemiDynamicWiDiD

widid = SemiDynamicWiDiD(
    similarity_threshold=0.78,
    historical_update_threshold=5,
    history_window=10000,
    min_cluster_fraction=0.005,
    max_cluster_age=8,
)

# Each row is one contextual embedding for the target word in that period.
labels_1850 = widid.partial_fit(vectors_1850, period=1850)
labels_1860 = widid.partial_fit(vectors_1860, period=1860)
labels_1870 = widid.partial_fit(vectors_1870, period=1870)

scores = widid.score_shift(
    past_periods=[1850, 1860],
    current_periods=[1870],
)

print(scores.jsd, scores.pdis, scores.pdiv)
print(widid.snapshot())
```

## Main parameters

- `similarity_threshold`: cosine similarity required to join an existing cluster.
- `historical_update_threshold`: number of `partial_fit()` batch updates before historical refresh.
- `history_window`: optional maximum number of retained historical occurrences.
- `min_cluster_fraction`: trims clusters smaller than this fraction of retained embeddings.
- `max_cluster_age`: trims clusters not updated for this many periods.

## Relation to the paper

The implementation follows the paper's WiDiD stages:

- document selection and embedding extraction happen outside the class;
- clustering uses APP-style incremental clustering: old clusters are packed into centroids, Affinity Propagation is run on old centroids plus new vectors, and old clusters are unpacked into the updated clustering;
- cluster refinement is implemented with minimum-size and age pruning;
- semantic shift is measured with JSD, PDIS, and PDIV.

The new semi-dynamic step is `historical_refresh()`. It reopens retained history after the threshold, re-clusters it with Affinity Propagation, then reconciles the new grouping with stable cluster IDs by centroid similarity.

APP requires scikit-learn:

```bash
pip install scikit-learn
```

If scikit-learn is not installed in the active interpreter, the runner stops with a clear error instead of silently using a different clustering algorithm.

## Recommended setting

Start with:

```python
SemiDynamicWiDiD(
    similarity_threshold=0.75,
    historical_update_threshold=5,
    history_window=5000,
    min_cluster_fraction=0.002,
    max_cluster_age=None,
)
```

Then tune:

- lower `similarity_threshold` if too many small clusters appear;
- raise `similarity_threshold` if unrelated senses are merging;
- lower `historical_update_threshold` if old decisions become stale quickly;
- raise `historical_update_threshold` if processing speed matters more.

## Running on SemEval-style data

Use the experiment runner when you have the SemEval corpus folders locally:

```bash
python3 semeval_widid_experiment.py \
  --corpus1 /path/to/corpus1 \
  --corpus2 /path/to/corpus2 \
  --targets /path/to/targets.txt \
  --gold /path/to/truth/graded.txt \
  --output widid_comparison.csv \
  --embedding-backend bert \
  --bert-model bert-base-uncased \
  --historical-update-threshold 5 \
  --batches-per-corpus 5 \
  --ap-preference-quantile 50
```

The runner writes one row per target word and method:

- `widid`: original incremental WiDiD baseline without historical refresh;
- `semi_dynamic_widid`: the thresholded historical-refresh version;
- `embedding_backend`: `hash`, `bert`, or `doc2vec`;
- `jsd`, `pdis`, `pdiv`: semantic-shift scores;
- `clusters`: number of final sense clusters;
- `batches_c1`, `batches_c2`: number of sequential batches actually used;
- `updates_processed`: total number of C1 and C2 batch updates;
- `historical_refreshes`: how many historical refreshes the semi-dynamic method performed;
- `historical_update_threshold`: refresh interval measured in batch updates;
- `ap_preference_quantile`, `ap_damping`, `min_cluster_fraction`: clustering configuration saved with the result;
- `c1_only_clusters`, `c2_only_clusters`, `mixed_clusters`: cluster composition diagnostics;
- `gold`: gold SemEval score when supplied.

If `--gold` is supplied, the runner also prints Spearman correlation for each method and metric.

The runner supports three embedding backends:

- `hash`: lightweight local context vectors, useful only for fast debugging.
- `bert`: contextual token embeddings from Hugging Face Transformers.
- `doc2vec`: pseudo-contextual document/sequence embeddings from Gensim.

For BERT, install:

```bash
pip install torch transformers
```

English example:

```bash
python3 semeval_widid_experiment.py \
  --corpus1 /path/to/corpus1 \
  --corpus2 /path/to/corpus2 \
  --targets /path/to/targets.txt \
  --gold /path/to/truth/graded.txt \
  --output widid_bert_english.csv \
  --embedding-backend bert \
  --bert-model bert-base-uncased \
  --historical-update-threshold 5 \
  --batches-per-corpus 5 \
  --ap-preference-quantile 50
```

Latin example:

```bash
python3 semeval_widid_experiment.py \
  --corpus1 /path/to/corpus1 \
  --corpus2 /path/to/corpus2 \
  --targets /path/to/targets.txt \
  --gold /path/to/truth/graded.txt \
  --output widid_bert_latin.csv \
  --embedding-backend bert \
  --bert-model bert-base-multilingual-uncased \
  --historical-update-threshold 5 \
  --batches-per-corpus 5 \
  --ap-preference-quantile 50
```

For Doc2Vec, install:

```bash
pip install gensim
```

Doc2Vec example:

```bash
python3 semeval_widid_experiment.py \
  --corpus1 /path/to/corpus1 \
  --corpus2 /path/to/corpus2 \
  --targets /path/to/targets.txt \
  --gold /path/to/truth/graded.txt \
  --output widid_doc2vec.csv \
  --embedding-backend doc2vec \
  --doc2vec-vector-size 100 \
  --doc2vec-window 10 \
  --doc2vec-epochs 15 \
  --historical-update-threshold 5 \
  --batches-per-corpus 5 \
  --ap-preference-quantile 50
```

If every `jsd` value is `0`, APP is producing cluster distributions that are
too similar across C1 and C2, often because too many occurrences are landing in
mixed clusters. Raise:

```bash
--ap-preference-quantile 75
```

then try:

```bash
--ap-preference-quantile 90
```

and finally:

```bash
--ap-preference-quantile 95
```

Higher values usually create more clusters. Check the output columns
`c1_only_clusters`, `c2_only_clusters`, and `mixed_clusters` to confirm whether
the clustering is too coarse or too fragmented.

Use `--batches-per-corpus` to expose the semi-dynamic behavior. With the default
value `1`, the runner only processes `C1 -> C2`, so original WiDiD and
semi-dynamic WiDiD can be identical. With values like `5` or `10`, the runner
streams each corpus in smaller chunks:

```text
C1_1 -> C1_2 -> ... -> C1_n -> C2_1 -> C2_2 -> ... -> C2_n
```

The final score is still computed as all C1 batches versus all C2 batches.
With five batches per corpus there are ten updates in total: threshold `5`
refreshes twice, threshold `10` refreshes once, and a threshold above `10`
does not refresh during that run.
