"""
Ranks retrieved documents utilizing the TF-IDF vector space model.
"""

import math
from preprocess import preprocess


# ─────────────────────────────────────────────────────────────
# TF-IDF Ranker Class
# ─────────────────────────────────────────────────────────────

class TFIDFRanker:
    """
    Ranks documents via the classic TF-IDF formulation.

    Formula:
        score(q, d) = Σ_{t ∈ q}  tf_norm(t,d) × idf(t)

        tf_norm(t, d)  = log(1 + tf(t,d))
        idf(t)         = log( (N + 1) / (df(t) + 1) )   [Smoothed IDF]
    """

    def __init__(self, inverted_index):
        self.inv = inverted_index

    def score(self, query: str, candidate_doc_ids: set) -> list[tuple[int, float]]:
        """
        Calculates raw TF-IDF scores for candidate documents and returns a ranked list.

        Parameters
        ----------
        query            : User query string
        candidate_doc_ids: Set of matching document IDs isolated by the QueryProcessor

        Returns
        -------
        list of (doc_id, score) sorted in descending order based on score values
        """
        if not candidate_doc_ids:
            return []

        terms = preprocess(query)
        if not terms:
            return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

        N = self.inv.N
        scores: dict[int, float] = {doc_id: 0.0 for doc_id in candidate_doc_ids}

        for term in terms:
            if term not in self.inv.index:
                continue

            df = self.inv.doc_freq.get(term, 0)
            idf = math.log((N + 1) / (df + 1))   # Smoothed IDF formula

            postings = self.inv.index[term]
            for doc_id in candidate_doc_ids:
                if doc_id not in postings:
                    continue
                tf = postings[doc_id].get("tf", 0)
                tf_norm = math.log(1 + tf)
                scores[doc_id] += tf_norm * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def rank(self, query: str, candidate_doc_ids: set,
             top_k: int = 10, normalize: bool = True) -> list[tuple[int, float]]:
        """
        Returns the top_k scored and optionally normalized documents.

        """
        scored = self.score(query, candidate_doc_ids)
        if normalize and scored:
            max_score = scored[0][1]
            if max_score > 0:
                scored = [(doc_id, round(sc / max_score, 4)) for doc_id, sc in scored]
        return scored[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from document_builder import load_data, build_documents
    from index_builder import InvertedIndex, FieldIndex
    from query_processor import QueryProcessor

    print("Loading data source files...")
    df   = load_data()
    docs = build_documents(df)

    print("Assembling system indexes...")
    inv = InvertedIndex()
    inv.build(docs)
    fi = FieldIndex()
    fi.build(docs)

    qp    = QueryProcessor(inv, fi)
    tfidf = TFIDFRanker(inv)

    test_queries = [
        "messi final",
        "mbappe AND goal",
        "penalty AND quarter-finals",
        "team:Argentina round:Final",
        'referee:"Szymon Marciniak"',
    ]

    print("\n=== Executing TF-IDF Ranking Verification ===\n")
    for q in test_queries:
        candidates = qp.get_matched_doc_ids(q)
        ranked = tfidf.rank(q, candidates, top_k=3, normalize=True)
        print(f'Query: "{q}"')
        for rank, (doc_id, sc) in enumerate(ranked, 1):
            m = docs[doc_id]["metadata"]
            print(f"  Rank {rank} | DocID: {doc_id} | {m['home_team']} vs {m['away_team']} | {m['round']} | Normalized Score: {sc:.4f}")
        print()