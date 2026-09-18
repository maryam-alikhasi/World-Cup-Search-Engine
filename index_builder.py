from collections import defaultdict
from preprocess import preprocess


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = {}
        self.doc_ids = set()
        self.N = 0

    def build(self, documents):
        self.N = len(documents)
        self.doc_ids = set(documents.keys())

        for doc_id, doc in documents.items():
            text_content = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            tokens = preprocess(text_content)
            self.doc_lengths[doc_id] = len(tokens)

            for pos, term in enumerate(tokens):
                if doc_id not in self.index[term]:
                    self.index[term][doc_id] = {"tf": 0, "positions": []}
                    self.doc_freq[term] += 1

                self.index[term][doc_id]["tf"] += 1
                self.index[term][doc_id]["positions"].append(pos)

    def get_postings(self, term):
        cleaned_terms = preprocess(term)
        if not cleaned_terms:
            return {}

        if len(cleaned_terms) == 1:
            return self.index.get(cleaned_terms[0], {})

        common_doc_ids = None
        for t in cleaned_terms:
            docs_with_term = set(self.index.get(t, {}).keys())
            if common_doc_ids is None:
                common_doc_ids = docs_with_term
            else:
                common_doc_ids.intersection_update(docs_with_term)

        if not common_doc_ids:
            return {}

        combined_postings = {}
        for doc_id in common_doc_ids:
            total_tf = sum(self.index[t][doc_id]["tf"] for t in cleaned_terms if doc_id in self.index[t])
            combined_postings[doc_id] = {"tf": total_tf}

        return combined_postings

    def get_doc_ids(self, term):
        return set(self.get_postings(term).keys())


class FieldIndex:
    def __init__(self):
        self.fields = defaultdict(lambda: defaultdict(set))

    def build(self, documents):
        for doc_id, doc in documents.items():
            for field_name, value in doc.get("fields", {}).items():
                if not value:
                    continue

                tokens = preprocess(str(value))
                field_name_lower = field_name.lower()

                for token in tokens:
                    self.fields[field_name_lower][token].add(doc_id)

    def match(self, field, value):
        field = field.lower()
        value_tokens = preprocess(value)

        if field not in self.fields or not value_tokens:
            return set()

        result_docs = None
        for token in value_tokens:
            docs_with_token = self.fields[field].get(token, set())
            if result_docs is None:
                result_docs = docs_with_token.copy()
            else:
                result_docs.intersection_update(docs_with_token)

            if not result_docs:
                return set()

        return result_docs if result_docs else set()