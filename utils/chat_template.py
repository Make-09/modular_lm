"""
utils/chat_template.py
========================
Converts instruction-following JSONL rows into tokenised tensors
using MultilingualTokenizer's chat format.

Chat format (token IDs):
  [BOS] <|system|> ... <|end|> <|user|> ... <|end|> <|assistant|> ... <|end|> [EOS]

For language-model training we predict the ASSISTANT turn only.
Loss is computed only on assistant tokens (input/label masking).
"""

from __future__ import annotations
import json
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from tokenizer import MultilingualTokenizer


# ── System prompts per language ────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "en": (
        "You are a helpful, accurate, and concise assistant. "
        "Respond in the same language as the user."
    ),
    "ru": (
        "Ты полезный, точный и лаконичный ассистент. "
        "Отвечай на том же языке, на котором пишет пользователь."
    ),
    "kk": (
        "Сіз пайдалы, дәл және нақты көмекшісіз. "
        "Пайдаланушы қай тілде жазса, сол тілде жауап беріңіз."
    ),
}

DEFAULT_SYSTEM = (
    "You are a helpful assistant that speaks Kazakh, Russian, and English. "
    "Сіз қазақша, орысша және ағылшынша сөйлей аласыз. "
    "Вы говорите на казахском, русском и английском языках."
)

IGNORE_ID = -100   # PyTorch CrossEntropy ignores this label index


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    """
    Loads a JSONL file with rows:
      {"lang": "ru", "instruction": "...", "input": "...", "output": "..."}

    Tokenises each example as a complete chat turn.
    Labels are IGNORE_ID everywhere except the assistant's response.

    Args
    ----
    jsonl_path : path to .jsonl file
    tokenizer  : MultilingualTokenizer instance
    max_len    : maximum sequence length (truncates)
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: MultilingualTokenizer,
        max_len: int = 256,
    ):
        self.tok     = tokenizer
        self.max_len = max_len
        self.rows: List[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        print(f"[ChatDataset] Loaded {len(self.rows)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return self._encode_row(row)

    def _encode_row(self, row: dict):
        lang        = row.get("lang", "en")
        instruction = row.get("instruction", "")
        inp         = row.get("input", "")
        output      = row.get("output", "")

        # Build user message
        user_msg = instruction
        if inp:
            user_msg = f"{instruction}\n\n{inp}"

        system = SYSTEM_PROMPTS.get(lang, DEFAULT_SYSTEM)

        # ── Encode prefix (everything before assistant response) ───────────────
        prefix_ids = self.tok.encode_chat(
            turns=[("user", user_msg)],
            system=system,
            add_generation_prompt=True,   # ends with <|assistant|>
        )

        # ── Encode response ────────────────────────────────────────────────────
        response_ids = (
            self.tok.encode(output)
            + [self.tok._END_ID if hasattr(self.tok, "_END_ID") else 7]
            + [self.tok.eos_id]
        )

        full_ids = prefix_ids + response_ids

        # ── Truncate ───────────────────────────────────────────────────────────
        full_ids = full_ids[: self.max_len]

        # ── Labels: IGNORE_ID for prefix, real IDs for response ───────────────
        prefix_len = min(len(prefix_ids), self.max_len)
        labels = (
            [IGNORE_ID] * prefix_len
            + full_ids[prefix_len:]
        )
        labels = labels[: self.max_len]

        # ── Pad to same length ─────────────────────────────────────────────────
        pad_len   = self.max_len - len(full_ids)
        input_ids = full_ids + [self.tok.pad_id] * pad_len
        labels    = labels   + [IGNORE_ID]       * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels    = torch.stack([b["labels"]    for b in batch])
    return input_ids, labels
