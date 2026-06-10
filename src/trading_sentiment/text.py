import re
import string

_DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}


def clean_text(text: str, stopwords: set[str] | None = None) -> str:
    """Lowercase text, remove punctuation, normalize spaces, and drop common stopwords."""
    stopwords = _DEFAULT_STOPWORDS if stopwords is None else stopwords
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word for word in text.split() if word not in stopwords)


def combine_title_summary(title: str, summary: str | None = None) -> str:
    """Combine article title and summary into one model-ready text field."""
    return f"{title or ''} {summary or ''}".strip()
