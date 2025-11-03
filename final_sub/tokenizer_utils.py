"""Utilities for text preprocessing and tokenizer persistence."""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

_PUNCTUATION_RE = re.compile(rf"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TokenizerArtifacts:
    tokenizer: Tokenizer
    index_word: Dict[int, str]
    vocab_size: int


def clean_text(text: str) -> str:
    """Lower-case and strip punctuation/extra whitespace."""
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_tokenizer(texts: Iterable[str], *, vocab_size: int, oov_token: str = "<OOV>") -> TokenizerArtifacts:
    """Create a Keras tokenizer fitted on the provided texts."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)

    vocab_size_eff = min(vocab_size, len(tokenizer.word_index) + 1)
    index_word = {
        idx: word
        for word, idx in tokenizer.word_index.items()
        if idx < vocab_size_eff
    }
    index_word.setdefault(0, "<PAD>")
    if oov_token:
        index_word.setdefault(tokenizer.word_index.get(oov_token, 1), oov_token)

    return TokenizerArtifacts(tokenizer=tokenizer, index_word=index_word, vocab_size=vocab_size_eff)


def save_tokenizer_artifacts(artifacts: TokenizerArtifacts, path: Path) -> None:
    """Persist tokenizer configuration for fast reloads."""
    payload = {
        "tokenizer": artifacts.tokenizer.to_json(),
        "vocab_size": artifacts.vocab_size,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_tokenizer_artifacts(path: Path) -> TokenizerArtifacts:
    """Load tokenizer configuration from disk."""
    path = Path(path)
    payload = json.loads(path.read_text())
    tokenizer = tokenizer_from_json(payload["tokenizer"])
    vocab_size = int(payload["vocab_size"])

    index_word = {
        idx: word
        for word, idx in tokenizer.word_index.items()
        if idx < vocab_size
    }
    index_word.setdefault(0, "<PAD>")
    if tokenizer.oov_token:
        index_word.setdefault(tokenizer.word_index.get(tokenizer.oov_token, 1), tokenizer.oov_token)

    return TokenizerArtifacts(tokenizer=tokenizer, index_word=index_word, vocab_size=vocab_size)


def encode_prompt(prompt: str, tokenizer: Tokenizer, *, seq_len: int) -> np.ndarray:
    """Vectorize a prompt into a padded numpy array."""
    cleaned = clean_text(prompt)
    seq = tokenizer.texts_to_sequences([cleaned])[0]
    seq = seq[-seq_len:]
    return pad_sequences([seq], maxlen=seq_len, padding="pre", truncating="pre")


def sliding_windows(sequence: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized many-to-one windows for efficiency."""
    if sequence.ndim != 1:
        raise ValueError("Sequence must be 1-D array of token ids")
    if window <= 0:
        raise ValueError("Window length must be positive")
    if sequence.shape[0] <= window:
        raise ValueError("Sequence length must exceed window length")

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(sequence, window_shape=window)
    X = windows[:-1]
    y = sequence[window:]
    return X.astype(np.int32), y.astype(np.int32)


__all__ = [
    "TokenizerArtifacts",
    "build_tokenizer",
    "clean_text",
    "encode_prompt",
    "load_tokenizer_artifacts",
    "save_tokenizer_artifacts",
    "sliding_windows",
]
