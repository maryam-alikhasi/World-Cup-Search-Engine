"""
Usage:
    python main.py                  ← Interactive Search Mode
    python main.py --evaluate       ← Run Full Evaluation Suite
    python main.py --query "messi"  ← Execute a Direct Query
"""

import sys
import argparse

sys.path.insert(0, ".")

from document_builder import load_data, build_documents
from index_builder import InvertedIndex, FieldIndex
from query_processor import QueryProcessor
from ranking import TFIDFRanker
from evaluation import (
    build_relevance_judgments,
    run_evaluation,
    print_report,
    print_top_results,
)


# ─────────────────────────────────────────────────────────────
# Loading and Index Building
# ─────────────────────────────────────────────────────────────

def build_system():
    print("═" * 60)
    print("  FIFA World Cup Match Search Engine")
    print("═" * 60)

    print("\n[1/3] Loading data and building documents...")
    df   = load_data()
    docs = build_documents(df)
    print(f"      {len(docs)} documents built successfully.")

    print("[2/3] Building Inverted Index and Field Index...")
    inv = InvertedIndex()
    inv.build(docs)
    fi = FieldIndex()
    fi.build(docs)
    print(f"      {len(inv.index)} terms indexed.")

    print("[3/3] Initializing Query Processor and Rankers...")
    qp    = QueryProcessor(inv, fi)
    tfidf = TFIDFRanker(inv)


    print("\nSystem is ready.\n")
    return docs, inv, fi, qp, tfidf


# ─────────────────────────────────────────────────────────────
# Result Display
# ─────────────────────────────────────────────────────────────

def display_results(query: str, ranked: list, docs: dict, top_k: int = 10):
    """Displays search results in a neat tabular format."""
    if not ranked:
        print(f'\n  No results found for "{query}".\n')
        return

    print(f'\n  Search results for: "{query}"  ({len(ranked)} hits)\n')
    print(f"  {'Rank':<5} {'DocID':<7} {'Home Team':<20} {'Away Team':<20} {'Round':<20} {'Score':<8} {'Year':<6} {'TF-IDF':>7}")
    print("  " + "─" * 95)

    for rank, (doc_id, score) in enumerate(ranked[:top_k], 1):
        m = docs[doc_id]["metadata"]
        print(
            f"  {rank:<5} {doc_id:<7} {m['home_team']:<20} {m['away_team']:<20} "
            f"{m['round']:<20} {m.get('score',''):<8} {m.get('year',''):<6} {score:>7.4f}"
        )
    print()


# ─────────────────────────────────────────────────────────────
# Interactive Mode Settings & Loop
# ─────────────────────────────────────────────────────────────

HELP_TEXT = """
Search Guide & Instructions:
  ─────────────────────────────────────────────────────────────
  messi                         Simple keyword search
  mbappe AND goal               AND operator
  penalty OR shootout           OR operator
  messi NOT france              NOT operator
  team:Argentina                Field search
  round:Final                   Field search
  referee:"Szymon Marciniak"    Phrase search within a field
  team:Argentina AND round:Final  Combining fields and boolean logic

  Searchable Fields:
    team, home_team, away_team, round, stage,
    stadium, city, referee, captain, coach,
    player, score, year, host

  Special Commands:
    :help   ─ Display this guide
    :eval   ─ Run the full system evaluation suite
    :quit   ─ Exit system
  ─────────────────────────────────────────────────────────────
"""


def interactive_mode(docs, qp, tfidf, top_k=10):
    print(HELP_TEXT)
    while True:
        try:
            query = input("  Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting system.")
            break

        if not query:
            continue
        if query.lower() in (":quit", ":q", "exit", "quit"):
            print("  Exiting system.")
            break
        if query.lower() == ":help":
            print(HELP_TEXT)
            continue
        if query.lower() == ":eval":
            print("  Running evaluation suite...")
            judgments = build_relevance_judgments(docs)
            results   = run_evaluation(docs, qp, tfidf, judgments, top_k=top_k)
            print_report(results, top_k=top_k)
            continue

        candidates = qp.get_matched_doc_ids(query)
        ranked     = tfidf.rank(query, candidates, top_k=top_k)
        display_results(query, ranked, docs, top_k=top_k)

def main():
    parser = argparse.ArgumentParser(description="FIFA World Cup Search Engine")
    parser.add_argument("--evaluate", action="store_true", help="Run full system evaluation suite")
    parser.add_argument("--query", type=str, default="", help="Execute a direct search query")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to display")
    args = parser.parse_args()

    docs, inv, fi, qp, tfidf = build_system()

    # ── Evaluation Mode ──────────────────────────────────────
    if args.evaluate:
        print("Generating relevance judgments...")
        judgments = build_relevance_judgments(docs)
        print(f"Total Queries: {len(judgments)}")
        for qid, jdg in judgments.items():
            print(f"  {qid}: {jdg['description']} — {len(jdg['relevant_ids'])} relevant docs")

        print("\nRunning system evaluation (TF-IDF)...")
        results = run_evaluation(docs, qp, tfidf, judgments, top_k=args.top_k)
        print_report(results, top_k=args.top_k)
        print_top_results(results, docs, top_k=5)
        return

    # ── Direct Query Mode ────────────────────────────────────
    if args.query:
        candidates = qp.get_matched_doc_ids(args.query)
        ranked     = tfidf.rank(args.query, candidates, top_k=args.top_k)
        display_results(args.query, ranked, docs, top_k=args.top_k)
        return

    # ── Interactive Mode ─────────────────────────────────────
    interactive_mode(docs, qp, tfidf, top_k=args.top_k)


if __name__ == "__main__":
    main()