import re
import unicodedata

STOPWORDS = set("""
        a an the and or but if then else when at by for with about against
        between into through during before after above below to from up down
        in out on off over under again further once here there all any both
        each few more most other some such no nor not only same so than too
        very s t can will just don should now is are was were be been being
        have has had do does did this that these those i you he she it we they
        of as
""".split())

SPECIAL_CHAR_MAP = {
    "&rsquor;": "'",
    "&rsquo;": "'",
    "&lsquo;": "'",
    "&quot;": '"',
    "&amp;": "and",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "\xa0": " ",
}

PUNCT_RE = re.compile(r"[^\w\s'\-]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_special_chars(text):
    for k, v in SPECIAL_CHAR_MAP.items():
        text = text.replace(k, v)
    return text


def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_text(text):
    text = clean_special_chars(text)
    text = strip_accents(text)
    text = text.lower()
    return text


def remove_punctuation(text):
    return PUNCT_RE.sub(" ", text)


def tokenize(text):
    text = WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    return text.split(" ")


def remove_stopwords(tokens):
    return [t for t in tokens if t and t not in STOPWORDS]


def preprocess(text):
    text = normalize_text(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return tokens