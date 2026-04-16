"""
tokenizer/train_tokenizer.py  — ИСПРАВЛЕННАЯ ВЕРСИЯ
======================================================
Использует только РАБОЧИЕ датасеты (parquet, без loading scripts):

  KK : kz-transformers/multidomain-kazakh-dataset  (streaming, первые N строк)
  RU : wikimedia/wikipedia  20231101.ru
  EN : wikimedia/wikipedia  20231101.en

Использование:
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --vocab_size 32000 --samples 30000
"""

import argparse
import os
import tempfile
from pathlib import Path

import sentencepiece as spm

# ── Источники данных ───────────────────────────────────────────────────────────
# (dataset_name, config, field, n_samples, use_streaming)
SOURCES = [
    # Казахский — streaming чтобы не качать 24M строк
    ("kz-transformers/multidomain-kazakh-dataset", "default", "text", 30_000, True),
    # Русская Википедия
    ("wikimedia/wikipedia", "20231101.ru", "text", 30_000, False),
    # Английская Википедия
    ("wikimedia/wikipedia", "20231101.en", "text", 30_000, False),
]


def iter_texts(n_samples: int):
    """Загружает текст из трёх языков и отдаёт построчно."""
    from datasets import load_dataset

    for dataset_name, config, field, default_n, use_streaming in SOURCES:
        n = min(n_samples, default_n)
        print(f"  Loading {dataset_name} ({config}) — {n} samples …")
        try:
            if use_streaming:
                # Streaming: берём только первые n строк без скачивания всего датасета
                ds = load_dataset(
                    dataset_name, config,
                    split="train",
                    streaming=True,
                    trust_remote_code=False,
                )
                count = 0
                for row in ds:
                    text = row.get(field, "")
                    if text and len(text) > 20:
                        yield text
                        count += 1
                    if count >= n:
                        break
                print(f"    → {count} docs streamed")
            else:
                ds = load_dataset(
                    dataset_name, config,
                    split=f"train[:{n}]",
                    trust_remote_code=False,
                )
                count = 0
                for row in ds:
                    text = row.get(field, "")
                    if text and len(text) > 20:
                        yield text
                        count += 1
                print(f"    → {count} docs loaded")
        except Exception as e:
            print(f"  [WARN] Failed to load {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",    type=int,   default=32_000)
    parser.add_argument("--samples",       type=int,   default=30_000,
                        help="Max samples per language")
    parser.add_argument("--output_dir",    default="tokenizer")
    parser.add_argument("--char_coverage", type=float, default=0.9999)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "multilingual")

    print("=" * 60)
    print("Training multilingual SentencePiece tokenizer")
    print(f"  Vocab size     : {args.vocab_size}")
    print(f"  Samples/lang   : {args.samples}")
    print(f"  Output prefix  : {model_prefix}")
    print("=" * 60)

    # ── Собираем обучающий текст ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name
        n_lines = 0
        print("\nCollecting text …")
        for text in iter_texts(args.samples):
            # Пишем первые 512 символов каждого документа
            f.write(text[:512].replace("\n", " ") + "\n")
            n_lines += 1
            if n_lines % 10_000 == 0:
                print(f"  {n_lines} lines written …")

    print(f"\nTotal lines: {n_lines}")
    if n_lines < 1000:
        raise RuntimeError(
            "Слишком мало данных для обучения токенизатора. "
            "Проверь подключение к интернету."
        )

    # ── Обучаем SentencePiece BPE ──────────────────────────────────────────────
    special_syms = ",".join(
        ["[PAD]", "[BOS]", "[EOS]",
         "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    )

    print("\nTraining SentencePiece BPE …")
    spm.SentencePieceTrainer.Train(
        input              = tmp_path,
        model_prefix       = model_prefix,
        vocab_size         = args.vocab_size,
        model_type         = "bpe",
        character_coverage = args.char_coverage,
        pad_id             = 0,
        unk_id             = 1,
        bos_id             = 2,
        eos_id             = 3,
        pad_piece          = "[PAD]",
        unk_piece          = "[UNK]",
        bos_piece          = "[BOS]",
        eos_piece          = "[EOS]",
        user_defined_symbols      = special_syms,
        num_threads               = os.cpu_count() or 4,
        input_sentence_size       = 5_000_000,
        shuffle_input_sentence    = True,
    )

    os.unlink(tmp_path)

    print(f"\n✓ Tokenizer saved:")
    print(f"  {model_prefix}.model")
    print(f"  {model_prefix}.vocab")

    # ── Быстрая проверка ──────────────────────────────────────────────────────
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