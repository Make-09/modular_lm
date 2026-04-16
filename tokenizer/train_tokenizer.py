"""
tokenizer/train_tokenizer.py
==============================
Train a SentencePiece BPE tokenizer on Kazakh, Russian, and English text.

Steps
-----
1. Downloads small samples from public HuggingFace datasets.
2. Writes combined text to a temp file.
3. Trains SentencePiece BPE with vocab_size=32 000.
4. Saves  tokenizer/multilingual.model  and  tokenizer/multilingual.vocab

Usage
-----
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --vocab_size 16000 --samples 50000
"""

import argparse
import os
import tempfile
from pathlib import Path

import sentencepiece as spm

# ── Data sources ───────────────────────────────────────────────────────────────
# Each entry: (hf_dataset_name, config/subset, split, text_field, n_samples)
SOURCES = [
    # Kazakh — Wikipedia + общие тексты
    ("kz-transformers/multidomain-kazakh-dataset", "default", "train", "text", 40_000),
    # Russian — Wikipedia
    ("wikimedia/wikipedia", "20231101.ru",           "train", "text", 40_000),
    # English — Wikipedia
    ("wikimedia/wikipedia", "20231101.en",           "train", "text", 40_000),
]

# Fallback if a dataset fails — uses mc4
FALLBACK_SOURCES = [
    ("mc4", "kk", "train", "text", 30_000),
    ("mc4", "ru", "train", "text", 30_000),
    ("mc4", "en", "train", "text", 30_000),
]


def iter_texts(n_samples: int):
    """Yield raw text strings from all three languages."""
    from datasets import load_dataset, DownloadConfig

    dl_cfg = DownloadConfig(max_retries=3)

    for dataset_name, config, split, field, default_n in SOURCES:
        n = min(n_samples, default_n)
        print(f"  Loading {dataset_name} ({config}) — {n} samples …")
        try:
            ds = load_dataset(
                dataset_name, config,
                split=f"{split}[:{n}]",
                trust_remote_code=True,
                download_config=dl_cfg,
            )
            for row in ds:
                text = row.get(field, "")
                if text and len(text) > 20:
                    yield text
        except Exception as e:
            print(f"  [WARN] Failed to load {dataset_name}: {e}")
            # Try fallback
            lang = config[:2]
            for fb_name, fb_cfg, fb_split, fb_field, fb_n in FALLBACK_SOURCES:
                if fb_cfg == lang:
                    try:
                        fb_ds = load_dataset(
                            fb_name, fb_cfg,
                            split=f"{fb_split}[:{min(n, fb_n)}]",
                            trust_remote_code=True,
                        )
                        for row in fb_ds:
                            t = row.get(fb_field, "")
                            if t and len(t) > 20:
                                yield t
                    except Exception as e2:
                        print(f"  [WARN] Fallback also failed: {e2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",  type=int, default=32_000)
    parser.add_argument("--samples",     type=int, default=50_000,
                        help="Max samples per language")
    parser.add_argument("--output_dir",  default="tokenizer")
    parser.add_argument("--char_coverage", type=float, default=0.9999,
                        help="Higher = better coverage of Cyrillic + Latin")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "multilingual")

    print("=" * 60)
    print("Training multilingual SentencePiece tokenizer")
    print(f"  Vocab size     : {args.vocab_size}")
    print(f"  Samples/lang   : {args.samples}")
    print(f"  Output prefix  : {model_prefix}")
    print("=" * 60)

    # ── Collect training text ──────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        n_lines = 0
        print("\nCollecting text …")
        for text in iter_texts(args.samples):
            # Write first 512 chars of each doc (avoids huge lines)
            f.write(text[:512].replace("\n", " ") + "\n")
            n_lines += 1
            if n_lines % 10_000 == 0:
                print(f"  {n_lines} lines written …")

    print(f"\nTotal lines: {n_lines}")
    if n_lines < 1000:
        print("[WARN] Very few lines — tokenizer quality will be low.")

    # ── Train SentencePiece ────────────────────────────────────────────────────
    # We reserve 8 IDs for special tokens; SentencePiece gets the rest.
    # user_defined_symbols lets us embed chat tokens in the model.
    special_syms = ",".join(
        ["[PAD]", "[BOS]", "[EOS]",
         "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    )

    print("\nTraining SentencePiece BPE …")
    spm.SentencePieceTrainer.Train(
        input            = tmp_path,
        model_prefix     = model_prefix,
        vocab_size       = args.vocab_size,
        model_type       = "bpe",
        character_coverage = args.char_coverage,
        pad_id           = 0,
        unk_id           = 1,
        bos_id           = 2,
        eos_id           = 3,
        pad_piece        = "[PAD]",
        unk_piece        = "[UNK]",
        bos_piece        = "[BOS]",
        eos_piece        = "[EOS]",
        user_defined_symbols = special_syms,
        num_threads      = os.cpu_count() or 4,
        input_sentence_size = 5_000_000,
        shuffle_input_sentence = True,
    )

    os.unlink(tmp_path)

    print(f"\n✓ Tokenizer saved:")
    print(f"  {model_prefix}.model")
    print(f"  {model_prefix}.vocab")

    # ── Quick sanity check ────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    for sample in [
        "Сәлем, сіздің атыңыз кім?",
        "Привет, как дела?",
        "Hello, how are you?",
    ]:
        pieces = sp.EncodeAsPieces(sample)
        print(f"  {sample!r:40s} → {pieces}")

    print("\nDone!")


if __name__ == "__main__":
    main()
