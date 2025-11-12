import re
import uuid
import logging
from typing import List, Dict, Any
import numpy as np

_device = -1  # CPU only

# ---- Optional imports ----
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_SENTE = True
except Exception:
    SentenceTransformer = None
    _HAVE_SENTE = False
    logging.warning("[ai_sum] sentence-transformers not available")

try:
    from transformers import pipeline
    _HAVE_TRANSFORMERS = True
except Exception:
    pipeline = None
    _HAVE_TRANSFORMERS = False
    logging.warning("[ai_sum] transformers not available")

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:
    nx = None
    _HAVE_NX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAVE_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    _HAVE_SKLEARN = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    _HAVE_NLTK = True
except Exception:
    sent_tokenize = lambda s: s.split('. ')
    _HAVE_NLTK = False

# ---------------- text utils ----------------
def clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    if not text: return []
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# ---------------- Extractive summarizer ----------------
class ExtractiveSummarizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not _HAVE_SENTE:
            self.embedder = None
            logging.warning("[ai_sum] Extractive summarizer disabled")
            return
        self.embedder = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sents = split_sentences(clean_text(text))
        if not sents: return ""
        if not self.embedder or len(sents) <= num_sentences:
            return " ".join(sents[:num_sentences])
        embeds = self.embedder.encode(sents, convert_to_numpy=True, show_progress_bar=False)
        sim = np.matmul(embeds, embeds.T)
        np.fill_diagonal(sim, 0)
        if _HAVE_NX:
            graph = nx.from_numpy_array(sim)
            scores = nx.pagerank_numpy(graph)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top = [sents[i] for i, _ in ranked[:num_sentences]]
        else:
            avg_scores = sim.sum(axis=1)
            top_idx = np.argsort(avg_scores)[::-1][:num_sentences]
            top = [sents[i] for i in sorted(top_idx)]
        return " ".join(top)

# ---------------- Abstractive summarizer ----------------
class AbstractiveSummarizer:
    def __init__(self, model_name="t5-small", device=_device):
        if not _HAVE_TRANSFORMERS:
            self._pipe = None
            logging.warning("[ai_sum] Abstractive summarizer disabled")
            return
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _init_pipe(self):
        if self._pipe is None and _HAVE_TRANSFORMERS:
            self._pipe = pipeline("summarization", model=self.model_name, device=self.device)

    def summarize(self, text: str, max_length=120, min_length=30) -> str:
        text = clean_text(text)
        if not text or not _HAVE_TRANSFORMERS:
            return ""
        self._init_pipe()
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = []
        for chunk in chunks:
            try:
                out = self._pipe(chunk, max_length=max_length, min_length=min_length, truncation=True)
                summaries.append(out[0]['summary_text'])
            except Exception as e:
                logging.warning(f"[ai_sum] Abstractive summarization chunk failed: {e}")
        return " ".join(summaries)

# ---------------- Keywords ----------------
def extract_keywords_tfidf(text: str, top_n: int = 8) -> List[str]:
    text = clean_text(text)
    if not text or not _HAVE_SKLEARN: return []
    sents = split_sentences(text)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(sents if len(sents) > 1 else [text])
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(tfidf.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_n]
    return terms[top_idx].tolist()

# ---------------- NER ----------------
def extract_entities_transformers(text: str, model_name="dslim/distilbert-base-NER") -> List[Dict[str, Any]]:
    if not _HAVE_TRANSFORMERS:
        return []
    try:
        ner_pipe = pipeline("ner", model=model_name, device=_device, grouped_entities=True)
        res = ner_pipe(clean_text(text)[:5000])
        entities = [{"entity_group": e.get("entity_group"), "word": e.get("word"), "score": round(float(e.get("score",0)),3)} for e in res]
        return entities
    except Exception as e:
        logging.warning(f"[ai_sum] NER error: {e}")
        return []

# ---------------- Combined wrapper ----------------
class SummarizerModule:
    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def summarize_and_extract(self, text: str, doc_id=None) -> Dict[str, Any]:
        text = clean_text(text)
        if not text:
            return {"doc_id": doc_id or str(uuid.uuid4()), "title":"", "abstractive_summary":"", "extractive_summary":"", "keywords":[], "entities":[]}

        abstractive_summary = self.abstractive.summarize(text)
        extractive_summary = self.extractive.summarize(text)
        keywords = extract_keywords_tfidf(text)
        entities = extract_entities_transformers(text)
        title = split_sentences(text)[0][:120] if split_sentences(text) else "Untitled"

        return {
            "doc_id": doc_id or str(uuid.uuid4()),
            "title": title,
            "full_text": text,
            "abstractive_summary": abstractive_summary,
            "extractive_summary": extractive_summary,
            "keywords": keywords,
            "entities": entities
        }

def summarize_text_and_get_schema(text: str, doc_id=None) -> Dict[str, Any]:
    sm = SummarizerModule()
    return sm.summarize_and_extract(text, doc_id=doc_id)

# ---------------- quick test ----------------
if __name__ == "__main__":
    sample = "SpaceX is a private aerospace company founded by Elon Musk. It develops rockets and spacecraft for space travel."
    print(summarize_text_and_get_schema(sample))
