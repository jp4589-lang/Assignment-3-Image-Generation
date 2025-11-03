# app/embeddings.py
from __future__ import annotations
from typing import List
import numpy as np
import spacy

MODEL_NAME = "en_core_web_lg"   # large English model with vectors
nlp = spacy.load(MODEL_NAME)    # downloaded in step 3

def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n != 0.0 else 1e-12

def word_vector(word: str) -> np.ndarray:
    doc = nlp(word.strip())
    if not doc or len(doc) == 0:
        return np.zeros(nlp.vocab.vectors_length, dtype=float)
    tok = doc[0]
    if not tok.has_vector:
        return np.zeros(nlp.vocab.vectors_length, dtype=float)
    return tok.vector

def sentence_vector(text: str) -> np.ndarray:
    doc = nlp(text.strip())
    vecs = [t.vector for t in doc if t.has_vector]
    if vecs:
        return np.mean(vecs, axis=0)
    # fallback (SpaCy gives a doc vector, but use zeros if missing)
    return np.zeros(nlp.vocab.vectors_length, dtype=float)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (_safe_norm(a) * _safe_norm(b)))
