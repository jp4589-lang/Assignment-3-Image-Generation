# app/bigram_model.py
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional


class BigramModel:
    """
    Builds a simple bigram language model from a list of text documents
    and can sample/generate text one word at a time.
    """

    def __init__(
        self,
        corpus: List[str],
        frequency_threshold: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            corpus: list of strings to train on
            frequency_threshold: if provided, drop tokens that occur fewer than
                this many times in the corpus (helps de-noise tiny corpora)
            seed: random seed for reproducible sampling
        """
        if seed is not None:
            random.seed(seed)

        self._raw_corpus = corpus
        text = "\n".join(corpus)
        self.tokens: List[str] = self._simple_tokenizer(
            text, frequency_threshold=frequency_threshold
        )

        self.vocab: List[str] = sorted(set(self.tokens))
        self.bigram_probs: Dict[str, Dict[str, float]] = self._build_bigram_probs(self.tokens)

    # ---------- public API ----------

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """
        Generate space-separated text using learned bigram probabilities.

        If start_word isn't in the model, a random vocabulary word is chosen.
        """
        if length <= 0:
            return ""

        current = start_word.lower()
        if current not in self.bigram_probs:
            current = random.choice(self.vocab) if self.vocab else start_word.lower()

        out = [current]
        for _ in range(length - 1):
            next_dist = self.bigram_probs.get(current)
            if not next_dist:
                break
            words, weights = zip(*next_dist.items())
            current = random.choices(words, weights=weights, k=1)[0]
            out.append(current)
        return " ".join(out)

    # ---------- helpers ----------

    @staticmethod
    def _simple_tokenizer(text: str, frequency_threshold: Optional[int] = None) -> List[str]:
        """Lowercase, split on word characters; optionally drop very rare words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if frequency_threshold:
            counts = Counter(tokens)
            tokens = [t for t in tokens if counts[t] >= frequency_threshold]
        return tokens

    @staticmethod
    def _build_bigram_probs(words: List[str]) -> Dict[str, Dict[str, float]]:
        """Count unigrams/bigrams and convert to conditional probabilities P(next|current)."""
        if len(words) < 2:
            return {}

        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (w1, w2), c in bigram_counts.items():
            probs[w1][w2] = c / unigram_counts[w1]
        return probs
