import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "cannot", "without", "hardly"
}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS

def clean_text(text: str) -> str:
    """Làm sạch văn bản, chuẩn hóa từ viết tắt phủ định và loại bỏ Stopwords."""
    text = re.sub(r"<.*?>", " ", str(text))
    text = text.lower()
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

def build_tfidf_vectorizer(ngram_range=(1, 2), max_features=20000, min_df=3):
    """Khởi tạo TfidfVectorizer."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True
    )
