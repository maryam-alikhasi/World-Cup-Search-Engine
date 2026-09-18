"""
Evaluation of the World Cup Information Retrieval System.

Includes:
  - 10 evaluation queries with manual relevance judgments (silver standard)
  - Metrics: Precision@K, Recall@K, F1@K, MAP, NDCG@K
"""

import math
import sys
sys.path.insert(0, ".")

from document_builder import load_data, build_documents
from index_builder import InvertedIndex, FieldIndex
from query_processor import QueryProcessor
from ranking import TFIDFRanker


# ─────────────────────────────────────────────────────────────
# Relevance Judgments Construction
# ─────────────────────────────────────────────────────────────

def build_relevance_judgments(docs: dict) -> dict:
    """
    Programmatically determines relevant documents for each query.

    Method: Inspects document text or metadata fields for each criterion.
    This approach serves as an automated silver standard, treated as
    manual ground truth for this project.

    Returns
    -------
    dict  →  { query_id: {"query": str, "relevant_ids": set} }
    """
    judgments = {}

    # Q1
    relevant = set()
    for doc_id, doc in docs.items():
        text = doc["text"].lower()
        fields = doc.get("fields", {})
        player_field = fields.get("player", "").lower()
        if "messi" in text and (
            "argentina goals: lionel messi" in text
            or "argentina goals:" in text and "messi" in text and "goals" in text
        ):
            relevant.add(doc_id)
    judgments["Q1"] = {
        "query": "messi",
        "description": "Matches where Lionel Messi scored a goal",
        "relevant_ids": relevant,
    }

    # Q2
    relevant = set()
    for doc_id, doc in docs.items():
        text = doc["text"].lower()
        if "penalty shootout decided the match" in text:
            relevant.add(doc_id)
    judgments["Q2"] = {
        "query": "penalty shootout",
        "description": "Matches decided by a penalty shootout",
        "relevant_ids": relevant,
    }

    # Q3
    relevant = set()
    for doc_id, doc in docs.items():
        meta = doc["metadata"]
        if meta.get("round", "").lower() == "final":
            relevant.add(doc_id)
    judgments["Q3"] = {
        "query": "round:Final",
        "description": "Final stage matches",
        "relevant_ids": relevant,
    }

    # Q4
    relevant = set()
    for doc_id, doc in docs.items():
        meta = doc["metadata"]
        teams = {meta.get("home_team", "").lower(), meta.get("away_team", "").lower()}
        round_ = meta.get("round", "").lower()
        if "argentina" in teams and round_ == "final":
            relevant.add(doc_id)
    judgments["Q4"] = {
        "query": "team:Argentina round:Final",
        "description": "Argentina matches in the Final stage",
        "relevant_ids": relevant,
    }

    # Q5
    relevant = set()
    for doc_id, doc in docs.items():
        text = doc["text"].lower()
        if "extra time goal" in text:
            relevant.add(doc_id)
    judgments["Q5"] = {
        "query": "extra time goal",
        "description": "Matches with goals scored during extra time",
        "relevant_ids": relevant,
    }

    # Q6
    relevant = set()
    for doc_id, doc in docs.items():
        meta = doc["metadata"]
        if "marciniak" in meta.get("referee", "").lower():
            relevant.add(doc_id)
    judgments["Q6"] = {
        "query": 'referee:"Szymon Marciniak"',
        "description": "Matches refereed by Szymon Marciniak",
        "relevant_ids": relevant,
    }

    # Q7
    relevant = set()
    for doc_id, doc in docs.items():
        text = doc["text"].lower()
        if "own goal" in text and ("goals:" in text):
            relevant.add(doc_id)
    judgments["Q7"] = {
        "query": "own goal",
        "description": "Matches featuring an own goal",
        "relevant_ids": relevant,
    }

    # Q8
    relevant = set()
    for doc_id, doc in docs.items():
        meta = doc["metadata"]
        teams = {meta.get("home_team", "").lower(), meta.get("away_team", "").lower()}
        round_ = meta.get("round", "").lower()
        if "morocco" in teams and "semi" in round_:
            relevant.add(doc_id)
    judgments["Q8"] = {
        "query": "round:Semi-finals team:Morocco",
        "description": "Morocco matches in the Semi-finals",
        "relevant_ids": relevant,
    }

    # Q9
    relevant = set()
    for doc_id, doc in docs.items():
        meta = doc["metadata"]
        if meta.get("score") == "0-0":
            relevant.add(doc_id)
    judgments["Q9"] = {
        "query": "0-0",
        "description": "Matches that ended in a 0-0 draw",
        "relevant_ids": relevant,
    }

    # Q10
    relevant = set()
    for doc_id, doc in docs.items():
        text = doc["text"].lower()
        if "red card" in text and "red cards:" in text:
            relevant.add(doc_id)
    judgments["Q10"] = {
        "query": "red card",
        "description": "Matches where a red card was given",
        "relevant_ids": relevant,
    }

    return judgments


# ─────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────

def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """P@K: Proportion of retrieved documents that are relevant in top K results"""
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """R@K: Proportion of relevant documents successfully retrieved in top K results"""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def f1_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """F1@K: Harmonic mean of P@K and R@K"""
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Average Precision for a single query.
    AP = (1/|R|) × Σ P@k × rel(k)
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for k, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant_ids:
            hits += 1
            ap_sum += hits / k
    return ap_sum / len(relevant_ids)


def mean_average_precision(results: list[tuple[list[int], set[int]]]) -> float:
    """MAP: Mean of AP values across all evaluation queries"""
    if not results:
        return 0.0
    return sum(average_precision(r, rel) for r, rel in results) / len(results)


def dcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """DCG@K using binary relevance (0 or 1)"""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K"""
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ─────────────────────────────────────────────────────────────
# Evaluation Execution
# ─────────────────────────────────────────────────────────────

def run_evaluation(docs, qp, ranker, judgments, top_k=10):
    """
    Runs a full system evaluation across all defined queries.

    Returns
    -------
    list of dict containing evaluation metrics for each query
    """
    results = []

    for qid, jdg in judgments.items():
        query       = jdg["query"]
        description = jdg["description"]
        relevant    = jdg["relevant_ids"]

        candidates = qp.get_matched_doc_ids(query)
        ranked_list = ranker.rank(query, candidates, top_k=top_k)
        ranked_ids  = [doc_id for doc_id, _ in ranked_list]

        p_k   = precision_at_k(ranked_ids, relevant, top_k)
        r_k   = recall_at_k(ranked_ids, relevant, top_k)
        f1_k  = f1_at_k(ranked_ids, relevant, top_k)
        ap    = average_precision(ranked_ids, relevant)
        ndcg  = ndcg_at_k(ranked_ids, relevant, top_k)

        results.append({
            "qid":          qid,
            "query":        query,
            "description":  description,
            "num_relevant": len(relevant),
            "num_retrieved":len(candidates),
            "ranked_ids":   ranked_ids,
            "P@K":          p_k,
            "R@K":          r_k,
            "F1@K":         f1_k,
            "AP":           ap,
            "NDCG@K":       ndcg,
        })

    return results


def print_report(results: list[dict], top_k: int = 10):
    """Prints the evaluation report in a tabular format"""
    sep = "─" * 110

    print(f"\n{'═'*110}")
    print(f"  World Cup IR System Evaluation Report   (K = {top_k})")
    print(f"{'═'*110}")
    print(f"{'QID':<5}  {'Query':<40}  {'|Rel|':>5}  {'|Ret|':>5}  {'P@K':>6}  {'R@K':>6}  {'F1@K':>6}  {'AP':>6}  {'NDCG':>6}")
    print(sep)

    for r in results:
        print(
            f"{r['qid']:<5}  "
            f"{r['query']:<40}  "
            f"{r['num_relevant']:>5}  "
            f"{r['num_retrieved']:>5}  "
            f"{r['P@K']:>6.3f}  "
            f"{r['R@K']:>6.3f}  "
            f"{r['F1@K']:>6.3f}  "
            f"{r['AP']:>6.3f}  "
            f"{r['NDCG@K']:>6.3f}"
        )

    print(sep)

    # Compute Averages
    n = len(results)
    avg_p    = sum(r["P@K"]    for r in results) / n
    avg_r    = sum(r["R@K"]    for r in results) / n
    avg_f1   = sum(r["F1@K"]   for r in results) / n
    map_val  = sum(r["AP"]     for r in results) / n
    avg_ndcg = sum(r["NDCG@K"] for r in results) / n

    print(
        f"{'Avg':<5}  "
        f"{'─'*40}  "
        f"{'─':>5}  {'─':>5}  "
        f"{avg_p:>6.3f}  "
        f"{avg_r:>6.3f}  "
        f"{avg_f1:>6.3f}  "
        f"{map_val:>6.3f}  "
        f"{avg_ndcg:>6.3f}"
    )
    print(f"{'═'*110}\n")
    print(f"  MAP (Mean Average Precision) = {map_val:.4f}")
    print(f"  Mean NDCG@{top_k}               = {avg_ndcg:.4f}")
    print(f"  Mean P@{top_k}                  = {avg_p:.4f}")
    print(f"  Mean R@{top_k}                  = {avg_r:.4f}")
    print(f"{'═'*110}\n")


def print_top_results(results: list[dict], docs: dict, top_k: int = 10):
    """Displays the top K retrieved results for each query"""
    print(f"\n{'═'*90}")
    print("  Top Retrieved Results per Query")
    print(f"{'═'*90}\n")

    for r in results:
        print(f"[{r['qid']}] {r['description']}")
        print(f"  Query: \"{r['query']}\"  |  Relevant: {r['num_relevant']}  |  Retrieved: {r['num_retrieved']}")
        print(f"  P@{top_k}={r['P@K']:.3f}  R@{top_k}={r['R@K']:.3f}  AP={r['AP']:.3f}  NDCG={r['NDCG@K']:.3f}")
        for rank, doc_id in enumerate(r["ranked_ids"][:top_k], 1):
            m = docs[doc_id]["metadata"]
            rel_mark = "✓" if doc_id in results[0]["ranked_ids"] else " "
            # Displaying match components from metadata
            print(
                f"    Rank {rank} | DocID:{doc_id:3d} | "
                f"{m['home_team']} vs {m['away_team']} | "
                f"{m['round']} | {m.get('year', '')}"
            )
        print()


if __name__ == "__main__":
    TOP_K = 10

    print("Loading dataset...")
    df   = load_data()
    docs = build_documents(df)

    print("Building indices...")
    inv = InvertedIndex()
    inv.build(docs)
    fi = FieldIndex()
    fi.build(docs)

    qp    = QueryProcessor(inv, fi)
    ranker = TFIDFRanker(inv)

    print("Generating relevance judgments...")
    judgments = build_relevance_judgments(docs)

    print(f"\nTotal Evaluation Queries: {len(judgments)}")
    for qid, jdg in judgments.items():
        print(f"  {qid}: {jdg['description']} — {len(jdg['relevant_ids'])} relevant docs")

    print("\nRunning system evaluation...")
    results = run_evaluation(docs, qp, ranker, judgments, top_k=TOP_K)

    print_report(results, top_k=TOP_K)
    print_top_results(results, docs, top_k=10)