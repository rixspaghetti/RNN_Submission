"""Train an LSTM-based next-word predictor on WikiText-2 with GloVe embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import json
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers

from final_sub.tokenizer_utils import (
    build_tokenizer,
    clean_text,
    save_tokenizer_artifacts,
    sliding_windows,
)

DEFAULT_SEQ_LEN = 10
DEFAULT_VOCAB = 20_000
DEFAULT_EMBED_DIM = 100
TOKENIZER_ARTIFACT = Path("artifacts/tokenizer.json")
MODEL_PATH = Path("best_wikitext2_lstm.keras")


def join_lines(dataset_split) -> str:
    texts = [t.strip() for t in dataset_split["text"] if t and t.strip()]
    return "\n".join(texts)


def load_glove_embeddings(glove_path: Path, vocab_size: int, tokenizer_word_index: dict) -> np.ndarray:
    embedding_matrix = np.zeros((vocab_size, DEFAULT_EMBED_DIM), dtype=np.float32)
    if not glove_path.exists():
        raise FileNotFoundError(
            f"Expected GloVe vectors at {glove_path}. See README for download instructions."
        )
    glove_index = {}
    with glove_path.open("r", encoding="utf8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if not parts:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            glove_index[word] = vec
    for word, idx in tokenizer_word_index.items():
        if idx >= vocab_size:
            continue
        vec = glove_index.get(word)
        if vec is not None and vec.shape[0] == DEFAULT_EMBED_DIM:
            embedding_matrix[idx] = vec
    return embedding_matrix


def prepare_datasets(seq_len: int, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, int]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text_train = clean_text(join_lines(dataset["train"]))
    text_val = clean_text(join_lines(dataset["validation"]))

    artifacts = build_tokenizer([text_train], vocab_size=vocab_size)
    save_tokenizer_artifacts(artifacts, TOKENIZER_ARTIFACT)

    seq_train = np.asarray(artifacts.tokenizer.texts_to_sequences([text_train])[0], dtype=np.int32)
    seq_val = np.asarray(artifacts.tokenizer.texts_to_sequences([text_val])[0], dtype=np.int32)

    X_train, y_train = sliding_windows(seq_train, seq_len)
    X_val, y_val = sliding_windows(seq_val, seq_len)

    return X_train, y_train, X_val, y_val, artifacts.tokenizer.word_index, artifacts.vocab_size


def build_model(vocab_size: int, embedding_matrix: np.ndarray, seq_len: int) -> keras.Model:
    embedding_layer = layers.Embedding(
        input_dim=vocab_size,
        output_dim=DEFAULT_EMBED_DIM,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=True,
        mask_zero=False,
        name="embedding",
    )

    model = keras.Sequential(
        [
            embedding_layer,
            layers.SpatialDropout1D(0.15, name="spatial_dropout"),
            layers.Bidirectional(
                layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                name="bilstm",
            ),
            layers.LayerNormalization(name="layer_norm"),
            layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2, name="lstm"),
            layers.Dense(256, activation="relu", name="dense_relu"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(vocab_size, activation="softmax", name="softmax"),
        ]
    )

    model.build(input_shape=(None, seq_len))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sliding window length")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB, help="Maximum vocabulary size")
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=Path("glove_6B/glove.6B.100d.txt"),
        help="Location of glove.6B.100d.txt",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, word_index, vocab_size_eff = prepare_datasets(
        seq_len=args.seq_len, vocab_size=args.vocab_size
    )

    embedding_matrix = load_glove_embeddings(args.glove_path, vocab_size_eff, word_index)

    model = build_model(vocab_size_eff, embedding_matrix, args.seq_len)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_output),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(50_000)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_path = args.model_output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history.history, default=float))


if __name__ == "__main__":
    main()
