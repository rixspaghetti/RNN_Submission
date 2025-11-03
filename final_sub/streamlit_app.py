#!/usr/bin/env python3
"""Interactive Streamlit demo for the WikiText-2 LSTM next-word predictor."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from final_sub.tokenizer_utils import (
    build_tokenizer,
    clean_text,
    encode_prompt,
    load_tokenizer_artifacts,
    save_tokenizer_artifacts,
)

# ---------------------------------------------------------------------------
# Constants shared with the notebook
# ---------------------------------------------------------------------------
SEQ_LEN = 10
VOCAB_SIZE = 20_000
MODEL_PATH = Path("best_model.keras")
GLOVE_PATH = Path("glove_6B/glove.6B.100d.txt")
TOKENIZER_PATH = Path("artifacts/tokenizer.json")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def join_lines(dataset_split) -> str:
    texts = [t.strip() for t in dataset_split["text"] if t and t.strip()]
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model file 'best_wikitext2_lstm.keras' is missing. "
            "Run the notebook to train/export the model before launching the app."
        )
    model = keras.models.load_model(MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def load_tokenizer() -> Tuple[Tokenizer, Dict[int, str], int]:
    if TOKENIZER_PATH.exists():
        artifacts = load_tokenizer_artifacts(TOKENIZER_PATH)
        return artifacts.tokenizer, artifacts.index_word, artifacts.vocab_size

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'datasets'. Install it with 'pip install datasets'."
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text_train = join_lines(dataset["train"])
    clean_train = clean_text(text_train)

    artifacts = build_tokenizer([clean_train], vocab_size=VOCAB_SIZE)
    save_tokenizer_artifacts(artifacts, TOKENIZER_PATH)
    return artifacts.tokenizer, artifacts.index_word, artifacts.vocab_size


@st.cache_resource(show_spinner=False)
def load_glove_vectors() -> Dict[str, np.ndarray]:
    if not GLOVE_PATH.exists():
        raise FileNotFoundError(
            "Expected GloVe vectors at 'glove_6B/glove.6B.100d.txt'."
            " Download/extract glove.6B.100d.txt and place it inside the glove_6B directory."
        )

    vectors: Dict[str, np.ndarray] = {}
    with GLOVE_PATH.open("r", encoding="utf8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if not parts:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------
def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(temperature, 1e-3)
    logits = np.log(probs + 1e-9) / temperature
    adjusted = np.exp(logits - np.max(logits))
    adjusted /= adjusted.sum()
    return adjusted


def top_k_predictions(
    probs: np.ndarray,
    index_word: Dict[int, str],
    top_k: int,
) -> List[Tuple[str, float]]:
    top_k = max(1, min(top_k, probs.shape[0]))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        token = index_word.get(int(idx), "<UNK>")
        results.append((token, float(probs[int(idx)])))
    return results


def sample_next_word(
    probs: np.ndarray,
    index_word: Dict[int, str],
    forbidden: Tuple[int, ...] = (0,),
) -> str:
    vocab = np.arange(len(probs))
    safe_mask = np.ones_like(probs, dtype=bool)
    if forbidden:
        safe_mask[list(forbidden)] = False
    safe_probs = probs * safe_mask
    if safe_probs.sum() <= 0:
        safe_probs = probs
    safe_probs = safe_probs / safe_probs.sum()
    token_id = np.random.choice(vocab, p=safe_probs)
    return index_word.get(int(token_id), "<UNK>")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Context-Aware Next-Word Suggestion",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 WikiText-2 Next-Word Studio")
st.caption(
    "Explore the pretrained LSTM that powers context-aware autocomplete. "
    "Type a sentence, tweak decoding settings, and visualize the confidence behind every suggestion."
)

# Delay loading heavy assets until needed
with st.spinner("Loading model..."):
    model = load_model()

with st.spinner("Loading tokenizer (first run may take ~1-2 minutes)..."):
    tokenizer, index_word, vocab_size_eff = load_tokenizer()


# Sidebar controls
with st.sidebar:
    st.header("Decode Controls")
    top_k = st.slider("Top-k suggestions", min_value=3, max_value=15, value=5, step=1)
    temperature = st.slider(
        "Temperature", min_value=0.3, max_value=1.5, value=0.9, step=0.1,
        help="Higher temperature = more exploratory suggestions"
    )
    continuation_words = st.slider(
        "Auto-complete length", min_value=1, max_value=25, value=10, step=1,
        help="Words to sample when generating a full continuation"
    )
    st.markdown("---")
    st.subheader("Model Snapshot")
    st.metric("Sequence length", SEQ_LEN)
    st.metric("Vocabulary size", vocab_size_eff)
    st.metric("Embedding dim", 100)


def ensure_session_defaults() -> None:
    if "prompt" not in st.session_state:
        st.session_state.prompt = "The history of the United"
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = st.session_state.prompt
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = None
    if "glove_ready" not in st.session_state:
        st.session_state.glove_ready = False


ensure_session_defaults()


def append_word_to_prompt(word: str) -> None:
    prompt = st.session_state.get("prompt", "").strip()
    new_prompt = (prompt + " " + word).strip()
    st.session_state.prompt = new_prompt
    st.session_state.prompt_input = new_prompt
    st.session_state.generated = []
    st.session_state.generated_text = None
    st.session_state.predictions = []
    st.experimental_rerun()


main_col, viz_col = st.columns((1.7, 1.3))

with main_col:
    st.subheader("Type a prompt")
    prompt_value = st.text_area(
        "", value=st.session_state.prompt, height=150, key="prompt_input"
    )
    st.session_state.prompt = prompt_value

    if st.button("Predict next word", type="primary"):
        prompt = st.session_state.prompt
        if not prompt.strip():
            st.warning("Please enter a prompt to get suggestions.")
        else:
            with st.spinner("Generating suggestions..."):
                encoded = encode_prompt(prompt, tokenizer, seq_len=SEQ_LEN)
                raw_probs = model(encoded, training=False)[0][:vocab_size_eff].numpy()
                tempered = apply_temperature(raw_probs, temperature)
                predictions = top_k_predictions(tempered, index_word, top_k)
            st.session_state.predictions = predictions

    if "predictions" in st.session_state:
        st.markdown("### Suggestions")
        preds = st.session_state.predictions
        if preds:
            for idx, (word, prob) in enumerate(preds, start=1):
                cols = st.columns((0.15, 0.55, 0.3))
                cols[0].markdown(f"**#{idx}**")
                cols[1].markdown(f"`{word}`")
                cols[2].write(f"{prob:.2%}")
                with cols[1]:
                    st.button(
                        f"Add → {word}",
                        key=f"add_{word}_{idx}",
                        on_click=append_word_to_prompt,
                        args=(word,),
                    )
        else:
            st.info("No predictions available yet.")

    if st.button("Generate continuation", key="generate"):
        prompt = st.session_state.prompt
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Sampling continuation..."):
                running_prompt = prompt
                generated_words: List[str] = []
                for _ in range(continuation_words):
                    encoded = encode_prompt(running_prompt, tokenizer, seq_len=SEQ_LEN)
                    raw_probs = model(encoded, training=False)[0][:vocab_size_eff].numpy()
                    tempered = apply_temperature(raw_probs, temperature)
                    next_word = sample_next_word(
                        tempered, index_word, forbidden=(0,)
                    )
                    generated_words.append(next_word)
                    running_prompt = running_prompt + " " + next_word
                st.session_state.generated = generated_words
                st.session_state.generated_text = running_prompt

    if st.session_state.get("generated_text"):
        st.markdown("### Auto-complete output")
        st.success(st.session_state.generated_text)

with viz_col:
    st.subheader("Confidence overview")
    if "predictions" in st.session_state and st.session_state.predictions:
        df = pd.DataFrame(
            {
                "word": [w for w, _ in st.session_state.predictions],
                "probability": [p for _, p in st.session_state.predictions],
            }
        )
        df.sort_values("probability", ascending=True, inplace=True)
        st.bar_chart(df, x="word", y="probability", height=320)
    else:
        st.info("Hit *Predict next word* to display the probability chart.")

    with st.expander("Tokenization details", expanded=False):
        st.write(
            "The tokenizer is cached to disk after the first build to avoid downloading "
            "WikiText-2 on every app restart."
        )
        st.json({
            "sequence_length": SEQ_LEN,
            "vocabulary_effective": vocab_size_eff,
            "oov_token": tokenizer.oov_token,
            "tokenizer_cache": str(TOKENIZER_PATH),
        })

    with st.expander("Word similarity explorer", expanded=False):
        st.write(
            "Compare semantic similarity using 100d GloVe vectors bundled with the project." 
            " Loading may take a few seconds the first time."
        )
        glove_ready = GLOVE_PATH.exists()
        if not glove_ready:
            st.warning("GloVe file not found. Download glove.6B.100d.txt to enable this tool.")
        else:
            if st.button("Load GloVe vectors", key="load_glove") or st.session_state.get("glove_ready"):
                st.session_state.glove_ready = True
                with st.spinner("Reading 100d embeddings..."):
                    glove_vectors = load_glove_vectors()
                word_a = st.text_input("Word A", value="king")
                word_b = st.text_input("Word B", value="queen")
                vec_a = glove_vectors.get(word_a.lower())
                vec_b = glove_vectors.get(word_b.lower())
                if vec_a is None or vec_b is None:
                    st.error("One of the words is missing from GloVe.")
                else:
                    sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
                    st.metric("Cosine similarity", f"{sim:.3f}")
            else:
                st.info("Click *Load GloVe vectors* to explore similarities.")


st.markdown("---")
st.markdown("#### Tips")
st.write(
    "- Use a **lower temperature** for deterministic, high-confidence autocompletion." 
    "\n- Increase **Top-k** to see more diverse suggestions." 
    "\n- Tap the **Add → word** buttons to iteratively craft a sentence."
)
