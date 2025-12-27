
# Learning-to-Rank for Search using Amazon Reviews

## Overview
This project demonstrates an Learning-to-Rank (LTR) pipeline
using public Amazon Reviews data. The goal is to construct a realistic ranking problem,
build strong baselines, train a learned ranker, and evaluate everything with **proper
ranking metrics**.

---

## Problem Formulation
We frame the task as a **search ranking problem**:

- **Query**: Review summary (short user intent proxy)
- **Document**: Review text
- **Relevance label**: Star rating (1–5), used as weak supervision
- **Ranking unit**: Documents are ranked *within the same query*

Because real search logs are unavailable, repeated review summaries are used as a
proxy for shared user intent.

---

## Dataset
- Source: Amazon Electronics Reviews (public Stanford SNAP dataset)
- Size (after filtering):
  - ~2100 documents
  - 181 unique queries
- Only queries with at least 5 documents are kept to ensure valid ranking groups.

---

## Data Splitting
- **Query-level split** (80% train / 20% test)
- Prevents information leakage across ranking groups

| Split | Queries | Documents |
|-----|--------|----------|
| Train | 144 | 1708 |
| Test  | 37  | 393  |

---

## Baseline: BM25
A classical information-retrieval baseline using lexical matching.

- Model: BM25 (rank-bm25)
- Metric: **NDCG@10**
- Evaluation: Query-level ranking

**Result:**
```
BM25 NDCG@10 ≈ 0.94
```

BM25 performs strongly due to high lexical overlap between review summaries and texts,
which is expected under this proxy setup.

---

## Learning-to-Rank Model
A simple but interpretable **pairwise learning-to-rank** model.

### Training Setup
- Pairwise data construction:
  - Positive document: rating ≥ 4
  - Negative document: rating ≤ 2
- Input: `query [SEP] document`
- Features: TF-IDF (unigrams + bigrams)
- Model: Logistic Regression

### Pairwise Classification Accuracy
```
Accuracy ≈ 0.50
```

This is expected due to weak supervision and balanced synthetic pairs.
Pairwise accuracy is **not** the primary evaluation metric.

---

## Ranking Evaluation (Learned Model)
The trained model is used as a scoring function to rank documents per query.

- Metric: **NDCG@10**
- Evaluation: Same query-level protocol as BM25

**Result:**
```
LTR NDCG@10 ≈ 0.94
```

---

## Results Summary

| Model | NDCG@10 |
|-----|--------|
| BM25 | ~0.94 |
| TF-IDF + Logistic (Pairwise LTR) | ~0.94 |

---

## Key Observations
- BM25 is a very strong baseline due to lexical overlap.
- The learned ranker matches BM25 performance under weak supervision.
- Pairwise classification accuracy is **not indicative of ranking quality**.
- Correct metric selection (NDCG) is critical for ranking problems.

---



## Reproducibility
All experiments are implemented in Python using:
- pandas, numpy
- scikit-learn
- rank-bm25

Random seeds are fixed where applicable.
