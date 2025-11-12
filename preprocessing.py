# preprocessing.py
"""
Preprocessing module for Research Navigator.
- Extracts text from PDF/DOCX
- Cleans text, splits into paragraphs and sentences
- Extracts entities using spaCy (en_core_web_sm)
- Integrates with ai_summarizer_combined.summarize_text_and_get_schema if available
- Falls back to TF-IDF keywords + extractive summary if AI summarizer not available
- Generates QnA: tries model-based generation if transformers available, else heuristic QnA
- Returns document_data dict ready to insert into DB
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Union, Any
import fitz  # PyMuPDF
from docx import Document
from nltk.tokenize import sent_tokenize
import spacy
from datetime import datetime
import uuid

# nltk punkt
import nltk
nltk.download("punkt", quiet=True)

# Try optional imports
_HAVE_AI_SUMMARIZER = False
try:
    from ai_summarizer_combined import summarize_text_and_get_schema  # optional helper
    _HAVE_AI_SUMMARIZER = True
except Exception:
    _HAVE_AI_SUMMARIZER = False

# Try optional transformers for QnA generation
_HAVE_TRANSFORMERS = False
_transformer_pipe = None
try:
    from transformers import pipeline
    # We'll try text2text-generation if available (models like valhalla/t5-small-qg)
    try:
        _transformer_pipe = pipeline("text2text-generation", model="valhalla/t5-small-qg")
        _HAVE_TRANSFORMERS = True
    except Exception:
        # fallback: try plain text-generation
        try:
            _transformer_pipe = pipeline("text-generation")
            _HAVE_TRANSFORMERS = True
        except Exception:
            _transformer_pipe = None
            _HAVE_TRANSFORMERS = False
except Exception:
    _HAVE_TRANSFORMERS = False
    _transformer_pipe = None

# Lazy-load spaCy model
_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        logging.info("Loading spaCy model en_core_web_sm...")
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger"])  # only ner
    return _NLP

# ---------------- Extraction ----------------
def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    try:
        with fitz.open(str(pdf_path)) as doc:
            return "".join(page.get_text("text") for page in doc).strip()
    except Exception as e:
        logging.error(f"[preproc] PDF extract error: {e}")
        return ""

def extract_text_from_docx(docx_path: Union[str, Path]) -> str:
    try:
        doc = Document(str(docx_path))
        return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        logging.error(f"[preproc] DOCX extract error: {e}")
        return ""

# ---------------- Cleaning / Sentences / Entities ----------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\t\r\f\v]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = text.strip()
    return text

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return sent_tokenize(text)

def extract_entities(text: str, max_len: int = 5000) -> List[Dict[str, str]]:
    try:
        nlp = get_nlp()
        shortened = (text or "")[:max_len]
        doc = nlp(shortened)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    except Exception as e:
        logging.error(f"[preproc] NER error: {e}")
        return []

def preprocess_paragraph(paragraph_text: str) -> Dict[str, Union[str, List]]:
    cleaned = clean_text(paragraph_text)
    sents = split_sentences(cleaned)
    return {
        "raw_text": paragraph_text,
        "cleaned_text": cleaned,
        "sentences": sents,
        "entities": extract_entities(cleaned)
    }

def preprocess_document_text(raw_text: str) -> List[Dict[str, Union[str, List]]]:
    paragraphs = [p for p in clean_text(raw_text).split("\n") if p.strip()]
    return [preprocess_paragraph(p) for p in paragraphs]

# ---------------- Keywords (TF-IDF fallback) ----------------
def extract_keywords_tfidf_from_text(text: str, top_n: int = 10, ngram_range=(1,2)) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        logging.warning("[preproc] sklearn not installed; skipping TF-IDF keywords.")
        return []

    text = clean_text(text)
    if not text:
        return []
    sents = split_sentences(text)
    try:
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=5000)
        X = tfidf.fit_transform(sents if len(sents) > 1 else [text])
        scores = X.sum(axis=0).A1
        terms = tfidf.get_feature_names_out()
        ranked_idx = scores.argsort()[::-1]
        top_terms = [terms[i] for i in ranked_idx[:top_n]]
        return top_terms
    except Exception as e:
        logging.error(f"[preproc] TF-IDF error: {e}")
        return []

# ---------------- Heuristic QnA generator (fallback) ----------------
def heuristic_generate_qna_from_summary(summary: str, keywords: List[str], entities: List[Union[str, Dict[str,Any]]], max_q=5) -> List[Dict[str, str]]:
    summary = (summary or "").strip()
    keywords = keywords or []
    # Normalize entities into simple strings
    ent_texts = []
    for e in entities or []:
        if isinstance(e, dict):
            ent_texts.append(e.get("text") or e.get("entity") or "")
        else:
            ent_texts.append(str(e))
    ent_texts = [x for x in ent_texts if x]

    qna = []
    # Q1 - what is it about
    qna.append({"q": "What is the paper about?", "a": summary or "Summary not available."})
    # Q2 - key topics
    if keywords:
        qna.append({"q": "What are the key topics / keywords?", "a": ", ".join(keywords[:8])})
    else:
        qna.append({"q": "What are the key topics / keywords?", "a": (", ".join(ent_texts[:8]) or "No keywords detected.")})
    # Q3 - contributions / findings (first 1-2 sentences)
    first_sents = (summary.split('. ')[:2])
    qna.append({"q": "What are the main contributions or findings?", "a": (". ".join(first_sents).strip() or "Not clear from the summary.")})
    # Q4 - methods hint
    methods_hint = ""
    if summary:
        lowered = summary.lower()
        verbs = ["propose", "introduce", "present", "develop", "use", "train", "evaluate", "measure", "apply", "simulate"]
        for v in verbs:
            if v in lowered:
                # find sentence containing verb
                parts = [s for s in summary.split('. ') if v in s.lower()]
                if parts:
                    methods_hint = parts[0].strip()
                    break
    if not methods_hint:
        methods_hint = "Method details are not explicitly detected — check the full text."
    qna.append({"q": "Which methods or approach are used?", "a": methods_hint})
    # Q5 - important entities / names
    qna.append({"q": "Which important names / entities to remember?", "a": (", ".join(list(dict.fromkeys(ent_texts))[:10]) or "No named entities detected.")})

    return qna[:max_q]

# ---------------- Optional model-based QnA (best-effort) ----------------
def model_generate_qna(summary: str, full_text: str, top_k: int = 5) -> List[Dict[str,str]]:
    """
    Best-effort QnA generation using transformers text2text pipeline if available.
    Input: summary (short), full_text (long)
    Output: list of {q,a}
    NOTE: This is optional and may fail if model not available; wrapped in try/except by caller.
    """
    if not _HAVE_TRANSFORMERS or _transformer_pipe is None:
        raise RuntimeError("Transformers text generation pipeline not available")

    # Use the pipeline to generate question-answer style outputs.
    # We'll build prompts of form: "generate question: <sentence> ==> answer: <sentence>"
    # Since different models behave differently, keep it simple and conservative.
    prompts = []
    # Use up to first 4 sentences of summary as seeds
    seeds = (summary or "").split('. ')
    seeds = [s for s in seeds if s.strip()][:4]
    if not seeds and full_text:
        # fallback to first sentences from full text
        seeds = (full_text or "").split('. ')[:4]

    for s in seeds:
        prompts.append(f"generate question and answer based on: {s.strip()}")

    qas = []
    try:
        for p in prompts[:top_k]:
            out = _transformer_pipe(p, max_length=128, do_sample=False)
            text_out = ""
            if isinstance(out, list) and out:
                # out may be a dict with 'generated_text' or 'text'
                if isinstance(out[0], dict):
                    text_out = out[0].get('generated_text') or out[0].get('text') or ""
                else:
                    text_out = str(out[0])
            text_out = (text_out or "").strip()
            if not text_out:
                continue
            # Try to split question/answer heuristically by '?' or 'answer:' etc.
            if '?' in text_out:
                parts = text_out.split('?')
                q = parts[0].strip() + '?'
                a = '?'.join(parts[1:]).strip()
                if not a:
                    a = "Answer not generated."
            elif "answer:" in text_out.lower():
                parts = re.split(r'answer\s*[:\-]\s*', text_out, flags=re.I)
                q = parts[0].strip()
                a = parts[1].strip() if len(parts) > 1 else "Answer not generated."
            else:
                # fallback: make the whole text an answer, question from seed
                q = (p[:150] + "...").strip()
                a = text_out
            qas.append({"q": q, "a": a})
    except Exception as e:
        logging.warning(f"[preproc] model_generate_qna failed: {e}")
    return qas[:top_k]

# ---------------- Main integration function ----------------
def process_uploaded_file(filepath: str, user_email: str, generate_qna_server_side: bool = False) -> Dict:
    """
    Main function used by app.py.
    Returns document_data dict with fields:
      doc_id, title, abstract, full_text (list of paragraphs with sentences/entities),
      entities (flattened), user_email, ingestion_date,
      abstractive_summary, extractive_summary, keywords (list), qna (list of {q,a})
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(filepath)
    elif ext == ".docx":
        raw_text = extract_text_from_docx(filepath)
    else:
        return {"status": "error", "message": "Unsupported format"}

    raw_text = clean_text(raw_text)
    processed_paragraphs = preprocess_document_text(raw_text)

    # old abstract: first sentence of first 3 paragraphs (backwards compatibility)
    abstract = " ".join([p["sentences"][0] for p in processed_paragraphs if p["sentences"]][:3]) if processed_paragraphs else ""

    doc_id = str(uuid.uuid4())
    title = Path(filepath).stem
    ingestion_date = datetime.utcnow()

    # defaults
    abstractive_summary = ""
    extractive_summary = ""
    keywords = []
    entities = [ent for p in processed_paragraphs for ent in p.get("entities", [])]

    # Try to use ai_summarizer_combined if available (preferred)
    if _HAVE_AI_SUMMARIZER:
        try:
            schema = summarize_text_and_get_schema(
                raw_text,
                abstractive_kwargs={"max_length": 200, "min_length": 50},
                extractive_sentences=5,
                top_k_keywords=12,
                ngram_range=(1,2),
                ner=True
            )
            # adopt schema values where present
            abstractive_summary = schema.get("abstractive_summary") or ""
            extractive_summary = schema.get("extractive_summary") or ""
            keywords = schema.get("keywords") or []
            # prefer entities from schema if provided
            if schema.get("entities"):
                entities = schema.get("entities")
        except Exception as e:
            logging.warning(f"[preproc] ai_summarizer failed: {e}")
            # fallback below

    # If still empty, fallback to extractive + TF-IDF keywords
    if not abstractive_summary:
        # extractive summary: first 5 sentences from document
        try:
            all_sents = []
            for p in processed_paragraphs:
                all_sents.extend(p.get("sentences", []))
            extractive_summary = extractive_summary or " ".join(all_sents[:5])
        except Exception:
            extractive_summary = extractive_summary or abstract

    if not keywords:
        keywords = extract_keywords_tfidf_from_text(raw_text, top_n=12)

    # QnA generation: try server-side model if requested AND available; else heuristic
    qna = []
    if generate_qna_server_side and _HAVE_TRANSFORMERS:
        try:
            qna = model_generate_qna(abstractive_summary or extractive_summary or abstract, raw_text, top_k=5)
        except Exception as e:
            logging.warning(f"[preproc] server-side QnA generation failed: {e}")
            qna = heuristic_generate_qna_from_summary(abstractive_summary or extractive_summary or abstract, keywords, entities, max_q=5)
    else:
        # heuristic QnA (fast and safe)
        qna = heuristic_generate_qna_from_summary(abstractive_summary or extractive_summary or abstract, keywords, entities, max_q=5)

    document_data = {
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "full_text": processed_paragraphs,
        "entities": entities,
        "user_email": user_email,
        "ingestion_date": ingestion_date,
        "abstractive_summary": abstractive_summary,
        "extractive_summary": extractive_summary,
        "keywords": keywords,
        "qna": qna
    }

    return document_data

# End of preprocessing.py
# this is my preprocessing.py update this also