"""
Processes user search queries and retrieves matching documents from the index.

Supports:
  1. Simple keyword search     → messi final
  2. Boolean AND/OR/NOT search → mbappe AND goal | penalty OR shootout | messi NOT france
  3. Structured field search   → team:Argentina round:Final | referee:"Szymon Marciniak"
  4. Field + Boolean mix       → team:Argentina AND round:Final
"""

import re
from preprocess import preprocess


# ─────────────────────────────────────────────────────────────
# Tokenizer / Parser Utilities
# ─────────────────────────────────────────────────────────────
# Pattern to capture field queries
FIELD_PATTERN = re.compile(r'(\w+):"([^"]+)"|(\w+):(\S+)')
BOOLEAN_OPS = {"and", "or", "not"}


def _split_tokens(query: str) -> list[str]:
    """
    Splits the raw query string into logical tokens.
    Quoted phrases are kept intact as single tokens.
    """
    tokens = []
    i = 0
    s = query.strip()
    while i < len(s):
        # Skip whitespace characters
        if s[i].isspace():
            i += 1
            continue
        # Field with a quoted value: referee:"John Doe"
        m = re.match(r'(\w+):"([^"]+)"', s[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        # Simple field without quotes: team:Argentina
        m = re.match(r'(\w+):(\S+)', s[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        # Quoted string literal without an explicit field: "extra time"
        if s[i] == '"':
            end = s.find('"', i + 1)
            if end == -1:
                end = len(s)
            tokens.append(s[i:end + 1])
            i = end + 1
            continue
        # Standard keyword token
        m = re.match(r'\S+', s[i:])
        if m:
            tokens.append(m.group(0))
            i += m.end()
            continue
        i += 1
    return tokens


def _is_bool_op(token: str) -> bool:
    return token.lower() in BOOLEAN_OPS


# ─────────────────────────────────────────────────────────────
# QueryProcessor Class
# ─────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Parses complex search strings and extracts document targets.

    Parameters
    ----------
    inverted_index : InvertedIndex
        The core inverted index containing document data.
    field_index : FieldIndex
        The specialized index for managing structured field metadata fields.
    """

    def __init__(self, inverted_index, field_index):
        self.inv = inverted_index
        self.fi = field_index

    # ── Primary API Methods ───────────────────────────────────

    def search(self, query: str) -> dict[int, dict]:
        """
        Parses a user query and returns a structured mapping of documents and match statistics.

        Returns
        -------
        dict  →  { doc_id: {"tf": <int>}, ... }
                 Matches the structural formatting expected by the TF-IDF Ranker.
        """
        query = query.strip()
        if not query:
            return {}

        tokens = _split_tokens(query)
        doc_ids = self._parse_expression(tokens)

        # Map document IDs into a postings structure containing aggregated term frequency details
        result = {}
        for doc_id in doc_ids:
            result[doc_id] = {"tf": self._total_tf_for_doc(query, doc_id)}
        return result

    def get_matched_doc_ids(self, query: str) -> set[int]:
        """Retrieves only the raw set of document IDs matching the query criteria."""
        return self._parse_expression(_split_tokens(query.strip()))

    # ── Expression Parser Logic ────────────────────────────────

    def _parse_expression(self, tokens: list[str]) -> set[int]:
        """
        Evaluates token arrays following the boolean operator precedence: NOT > AND > OR.
        Implements an iterative evaluation parser strategy.
        """
        # Split blocks by the OR operator to isolate processing components
        or_parts = self._split_by_op(tokens, "or")
        or_result = set()

        for part in or_parts:
            and_parts = self._split_by_op(part, "and")
            and_result = None

            for ap in and_parts:
                # Handle NOT operators which can modify terms (e.g., NOT france or messi NOT france)
                not_parts = self._split_by_op(ap, "not")

                if len(not_parts) == 1:
                    # No active NOT operator found in this block
                    if ap and ap[0].lower() == "not":
                        operand = self._eval_single(ap[1:])
                        res = self.inv.doc_ids - operand
                    else:
                        res = self._eval_single(ap)
                else:
                    # The leading segment represents the target positive matches
                    positive = self._eval_single(not_parts[0]) if not_parts[0] else self.inv.doc_ids
                    # Subtract all trailing terms impacted by sequential NOT statements
                    for not_part in not_parts[1:]:
                        positive = positive - self._eval_single(not_part)
                    res = positive

                if and_result is None:
                    and_result = res
                else:
                    and_result &= res

            if and_result:
                or_result |= and_result

        return or_result

    def _split_by_op(self, tokens: list[str], op: str) -> list[list[str]]:
        """Splits a list of tokens into distinct nested arrays around a specified boolean operator."""
        parts = []
        current = []
        for tok in tokens:
            if tok.lower() == op:
                parts.append(current)
                current = []
            else:
                current.append(tok)
        parts.append(current)
        return [p for p in parts if p]

    def _eval_single(self, tokens: list[str]) -> set[int]:
        """" evaluates a block without AND/OR/NOT """
        if not tokens:
            return set()

        result = None
        for tok in tokens:
            ids = self._resolve_token(tok)
            if not ids:
                # ignore if token doesn't have any result
                continue
            if result is None:
                result = ids
            else:
                result &= ids

        return result if result is not None else set()

    # ── Token Resolver Logic ───────────────────────────────────

    def _resolve_token(self, token: str) -> set[int]:
        """Resolves an isolated, single query token into a set of matching document IDs."""
        # Field queries (unchanged)
        m = re.match(r'(\w+):"([^"]+)"$', token)
        if m:
            return self.fi.match(m.group(1), m.group(2))

        m = re.match(r'(\w+):(.+)$', token)
        if m:
            return self.fi.match(m.group(1), m.group(2))

        # Quoted phrase
        if token.startswith('"') and token.endswith('"'):
            phrase = token[1:-1]
            return self._phrase_search(phrase)

        # Standard Keyword Term Search
        terms = preprocess(token)
        if not terms:
            return set()

        # For free-text queries, employ a UNION (OR) strategy rather than an INTERSECTION (AND)
        result = set()
        for t in terms:
            docs = set(self.inv.index.get(t, {}).keys())
            result = result.union(docs)  # Key modification: UNION replacing the previous AND logic

        return result

    def _phrase_search(self, phrase: str) -> set[int]:
        """
        Performs an exact phrase search utilizing positional data captured within the inverted index.
        Falls back to intersection tracking (AND) if positional tracking properties are missing.
        """
        terms = preprocess(phrase)
        if not terms:
            return set()
        if len(terms) == 1:
            return set(self.inv.index.get(terms[0], {}).keys())

        # Determine the initial intersection set of documents matching all target terms
        common = None
        for t in terms:
            docs = set(self.inv.index.get(t, {}).keys())
            common = docs if common is None else common & docs
        if not common:
            return set()

        # Validate incremental positional continuity across matches
        matched = set()
        for doc_id in common:
            positions = [
                self.inv.index[t][doc_id].get("positions", [])
                for t in terms
            ]
            if any(not p for p in positions):
                # Fallback directly to simple intersection if accurate tracking arrays are absent
                matched.add(doc_id)
                continue

            # Verify terms appear consecutively in sequential order
            first_positions = positions[0]
            for start in first_positions:
                if all(
                        (start + offset) in set(positions[offset])
                        for offset in range(1, len(terms))
                ):
                    matched.add(doc_id)
                    break
        return matched

    # ── Term Frequency Helper Methods ─────────────────────────

    def _total_tf_for_doc(self, query: str, doc_id: int) -> int:
        """
        Calculates the aggregated Term Frequency values across query terms for a given document.
        Used to seed the initial structural properties needed during relevance rank processing steps.
        """
        terms = preprocess(query)
        total = 0
        for t in terms:
            postings = self.inv.index.get(t, {})
            if doc_id in postings:
                total += postings[doc_id].get("tf", 0)
        return max(total, 1)


# ─────────────────────────────────────────────────────────────
# Isolated Testing Execution Block
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")

    from document_builder import load_data, build_documents
    from index_builder import InvertedIndex, FieldIndex

    print("Loading raw dataset records...")
    df = load_data()
    docs = build_documents(df)

    print("Constructing multi-tier indexes...")
    inv = InvertedIndex()
    inv.build(docs)

    fi = FieldIndex()
    fi.build(docs)

    qp = QueryProcessor(inv, fi)

    test_queries = [
        "messi",
        "mbappe goal",
        "messi AND final",
        "penalty OR shootout",
        "messi NOT france",
        "team:Argentina",
        "round:Final",
        'referee:"Szymon Marciniak"',
        "team:Argentina AND round:Final",
        "own goal",
        "extra time goal",
        "red card",
        "0-0",
    ]

    print("\n=== Executing Search System Verification Queries ===\n")
    for q in test_queries:
        result = qp.get_matched_doc_ids(q)
        print(f'  "{q}"  →  {len(result)} matching docs found | Sample doc_ids: {sorted(result)[:5]}')