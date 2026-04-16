"""
data/prepare_data.py
======================
Download and prepare multilingual instruction-following datasets
for Kazakh (KAZ), Russian (RUS), and English (ENG).

Datasets used
-------------
  KAZ : kz-transformers/multidomain-kazakh-dataset  (plain text — for pretraining)
        + translated from ru_turbo_alpaca via simple markers
  RUS : IlyaGusev/ru_turbo_alpaca                   (instruction, input, output)
  ENG : tatsu-lab/alpaca                             (instruction, input, output)

Output
------
  data/processed/train.jsonl   — all splits combined, shuffled
  data/processed/val.jsonl     — 5 % held-out
  data/processed/pretrain.txt  — plain text for Stage 1 (all languages)

Each JSONL line:
  {"lang": "ru", "instruction": "...", "input": "...", "output": "..."}

Usage
-----
    python data/prepare_data.py
    python data/prepare_data.py --max_per_lang 10000
"""

import argparse
import json
import os
import random
from typing import List, Dict

# ── Dataset specs ──────────────────────────────────────────────────────────────
# (hf_name, config, split, field_map, lang, max_samples)
# field_map: keys in output → source field names
INSTRUCTION_SOURCES = [
    # English Alpaca — original
    {
        "name": "tatsu-lab/alpaca",
        "config": None,
        "split": "train",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "lang": "en",
        "max": 20_000,
    },
    # Russian turbo-alpaca (GPT-4 translated)
    {
        "name": "IlyaGusev/ru_turbo_alpaca",
        "config": None,
        "split": "train",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "lang": "ru",
        "max": 20_000,
    },
    # Kazakh — if specific instruction dataset is unavailable we skip gracefully
    {
        "name": "kz-transformers/kazakh_instruct",
        "config": None,
        "split": "train",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "lang": "kk",
        "max": 10_000,
    },
]

PRETRAIN_SOURCES = [
    {"name": "kz-transformers/multidomain-kazakh-dataset", "config": "default",
     "split": "train", "field": "text", "lang": "kk", "max": 30_000},
    {"name": "wikimedia/wikipedia", "config": "20231101.ru",
     "split": "train", "field": "text", "lang": "ru", "max": 20_000},
    {"name": "wikimedia/wikipedia", "config": "20231101.en",
     "split": "train", "field": "text", "lang": "en", "max": 20_000},
]


def load_instruction_dataset(spec: Dict, max_samples: int) -> List[Dict]:
    from datasets import load_dataset
    n = min(spec["max"], max_samples)
    print(f"  [{spec['lang'].upper()}] {spec['name']} — {n} samples …")
    try:
        ds = load_dataset(
            spec["name"],
            spec["config"],
            split=f"{spec['split']}[:{n}]",
            trust_remote_code=True,
        )
        rows = []
        fm = spec["fields"]
        for row in ds:
            rows.append({
                "lang":        spec["lang"],
                "instruction": row.get(fm["instruction"], ""),
                "input":       row.get(fm.get("input", ""), ""),
                "output":      row.get(fm["output"], ""),
            })
        print(f"    → {len(rows)} examples loaded")
        return rows
    except Exception as e:
        print(f"    [WARN] Skipped — {e}")
        return []


def load_pretrain_text(spec: Dict, max_samples: int) -> List[str]:
    from datasets import load_dataset
    n = min(spec["max"], max_samples)
    print(f"  [{spec['lang'].upper()}] {spec['name']} pretrain — {n} docs …")
    try:
        ds = load_dataset(
            spec["name"],
            spec["config"],
            split=f"{spec['split']}[:{n}]",
            trust_remote_code=True,
        )
        texts = [row[spec["field"]] for row in ds if row.get(spec["field"])]
        print(f"    → {len(texts)} docs loaded")
        return texts
    except Exception as e:
        print(f"    [WARN] Skipped — {e}")
        return []


def split_train_val(data: List, val_ratio: float = 0.05):
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    return data[n_val:], data[:n_val]


def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} rows → {path}")


def save_text(texts: List[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n\n")
    print(f"  Saved {len(texts)} docs → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_lang", type=int, default=20_000,
                        help="Max instruction examples per language")
    parser.add_argument("--output_dir",   default="data/processed")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Preparing multilingual instruction dataset (KAZ / RUS / ENG)")
    print("=" * 60)

    # ── Instruction data ───────────────────────────────────────────────────────
    print("\n[1/2] Downloading instruction data …")
    all_rows: List[Dict] = []
    for spec in INSTRUCTION_SOURCES:
        rows = load_instruction_dataset(spec, args.max_per_lang)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(
            "No instruction data was downloaded. "
            "Check your internet connection or dataset names."
        )

    train, val = split_train_val(all_rows)
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val,   os.path.join(args.output_dir, "val.jsonl"))

    # ── Pretraining text ───────────────────────────────────────────────────────
    print("\n[2/2] Downloading pretraining text …")
    all_texts: List[str] = []
    for spec in PRETRAIN_SOURCES:
        texts = load_pretrain_text(spec, args.max_per_lang)
        all_texts.extend(texts)

    random.shuffle(all_texts)
    save_text(all_texts, os.path.join(args.output_dir, "pretrain.txt"))

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    from collections import Counter
    lang_counts = Counter(r["lang"] for r in all_rows)
    print("Language distribution (instruction data):")
    for lang, cnt in sorted(lang_counts.items()):
        print(f"  {lang.upper()} : {cnt:,}")
    print(f"\nTotal train : {len(train):,}")
    print(f"Total val   : {len(val):,}")
    print(f"Pretrain    : {len(all_texts):,} docs")
    print("=" * 60)
    print("\n✓ Data ready! Next steps:")
    print("  1. python tokenizer/train_tokenizer.py")
    print("  2. python stage1_train.py --data_path data/processed/pretrain.txt")
    print("  3. python stage2_memory.py --block1_ckpt checkpoints/block1.pt")
    print("  4. python stage3_fusion.py ...")
    print("  5. python finetune_chat.py ...")


if __name__ == "__main__":
    main()
