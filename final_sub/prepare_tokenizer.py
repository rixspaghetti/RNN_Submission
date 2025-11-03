"""CLI helper to build and persist the WikiText-2 tokenizer artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from final_sub.tokenizer_utils import build_tokenizer, clean_text, save_tokenizer_artifacts

DEFAULT_VOCAB = 20_000


def join_lines(dataset_split) -> str:
    texts = [t.strip() for t in dataset_split["text"] if t and t.strip()]
    return "\n".join(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB, help="Maximum vocabulary size")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tokenizer.json"),
        help="Where to store the tokenizer artifacts",
    )
    args = parser.parse_args()

    print("Downloading WikiText-2 (via datasets)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text_train = join_lines(dataset["train"])
    clean_train = clean_text(text_train)

    print("Fitting tokenizer...")
    artifacts = build_tokenizer([clean_train], vocab_size=args.vocab_size)

    print(f"Saving tokenizer to {args.output} ...")
    save_tokenizer_artifacts(artifacts, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
