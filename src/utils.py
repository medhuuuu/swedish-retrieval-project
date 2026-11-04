"""
Shared utility functions for the Swedish retrieval project.
Import with: from src.utils import *  (or adjust path as needed)
"""

from datasets import load_dataset, concatenate_datasets
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------- Dataset loading ----------

def load_swefaq():
    corpus = load_dataset("mteb/SweFaqRetrieval", "corpus")["test"]
    queries = load_dataset("mteb/SweFaqRetrieval", "queries")["test"]
    qrels = load_dataset("mteb/SweFaqRetrieval", "qrels")["test"]
    return corpus, queries, qrels


def load_swedn():
    splits = ["train", "validation", "test"]
    all_corpus, all_queries, all_qrels = [], [], []
    for split in splits:
        corpus_split = load_dataset("mteb/SwednRetrieval", "corpus")[split].map(lambda x: {"_id": f"{split}_{x['_id']}"})
        queries_split = load_dataset("mteb/SwednRetrieval", "queries")[split].map(lambda x: {"_id": f"{split}_{x['_id']}"})
        qrels_split = load_dataset("mteb/SwednRetrieval", "qrels")[split].map(lambda x: {
            "query-id": f"{split}_{x['query-id']}", "corpus-id": f"{split}_{x['corpus-id']}"
        })
        all_corpus.append(corpus_split); all_queries.append(queries_split); all_qrels.append(qrels_split)
    return concatenate_datasets(all_corpus), concatenate_datasets(all_queries), concatenate_datasets(all_qrels)


def build_qrels_dict(qrels):
    d = {}
    for row in qrels:
        d.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return d


# ---------- BM25 ----------

def build_bm25(corpus):
    corpus_ids = [doc["_id"] for doc in corpus]
    corpus_texts = [doc["text"] for doc in corpus]
    tokenized_corpus = [t.lower().split() for t in corpus_texts]
    return BM25Okapi(tokenized_corpus), corpus_ids, corpus_texts


# ---------- Chunking (long documents) ----------

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_chunked_corpus(corpus):
    chunked_ids, chunked_texts, parent_ids = [], [], []
    for doc in corpus:
        doc_id = doc["_id"]
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunked_ids.append(f"{doc_id}_chunk{i}")
            chunked_texts.append(chunk)
            parent_ids.append(doc_id)
    return chunked_ids, chunked_texts, parent_ids


# ---------- Dense retrieval ----------

def search_dense(query_text, model, corpus_embeddings, corpus_ids, top_k=10):
    query_emb = model.encode(["query: " + query_text], normalize_embeddings=True)
    scores = (corpus_embeddings @ query_emb.T).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [corpus_ids[i] for i in top_indices]


def dense_ranked_ids_chunked(query_text, model, chunk_embeddings, parent_ids, pool_size=100):
    query_emb = model.encode(["query: " + query_text], normalize_embeddings=True)
    scores = (chunk_embeddings @ query_emb.T).flatten()
    doc_best_score = {}
    for score, pid in zip(scores, parent_ids):
        if pid not in doc_best_score or score > doc_best_score[pid]:
            doc_best_score[pid] = score
    ranked = sorted(doc_best_score.items(), key=lambda x: x[1], reverse=True)[:pool_size]
    return [doc_id for doc_id, score in ranked]


# ---------- Hybrid (RRF) ----------

def reciprocal_rank_fusion(bm25_ranked_ids, dense_ranked_ids, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(dense_ranked_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in ranked]


def search_hybrid(query_text, bm25, corpus_ids, dense_embeddings, model, top_k=10, pool_size=100):
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = bm25_scores.argsort()[::-1][:pool_size]
    bm25_ranked_ids = [corpus_ids[i] for i in bm25_top_indices]
    dense_ranked_ids = search_dense(query_text, model, dense_embeddings, corpus_ids, top_k=pool_size)
    fused_ranking = reciprocal_rank_fusion(bm25_ranked_ids, dense_ranked_ids)
    return fused_ranking[:top_k]


def search_hybrid_chunked(query_text, bm25, corpus_ids, chunk_embeddings, parent_ids, model, top_k=10, pool_size=100):
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = bm25_scores.argsort()[::-1][:pool_size]
    bm25_ranked_ids = [corpus_ids[i] for i in bm25_top_indices]
    dense_ranked_ids = dense_ranked_ids_chunked(query_text, model, chunk_embeddings, parent_ids, pool_size)
    fused_ranking = reciprocal_rank_fusion(bm25_ranked_ids, dense_ranked_ids)
    return fused_ranking[:top_k]


# ---------- Evaluation metrics ----------

def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = set(retrieved_ids[:k])
    if len(relevant_ids) == 0:
        return None
    return len(top_k & relevant_ids) / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = sum(1/np.log2(i+2) for i, d in enumerate(retrieved_ids[:k]) if d in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1/np.log2(i+2) for i in range(ideal_hits))
    return dcg/idcg if idcg > 0 else None


def evaluate_retrieval(search_fn, queries, qrels_dict, k_values=[1, 3, 10]):
    results = {f"Recall@{k}": [] for k in k_values}
    results["nDCG@10"] = []
    for q in queries:
        qid = q["_id"]
        if qid not in qrels_dict:
            continue
        retrieved_ids = search_fn(q["text"])
        relevant_ids = qrels_dict[qid]
        for k in k_values:
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            if r is not None:
                results[f"Recall@{k}"].append(r)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, 10)
        if ndcg is not None:
            results["nDCG@10"].append(ndcg)
    return {m: np.mean(v) for m, v in results.items()}