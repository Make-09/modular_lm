"""
tokenizer/multilingual_tokenizer.py
=====================================
SentencePiece BPE wrapper with special chat tokens.

Special token layout (IDs 0-7 are reserved):
  [PAD]=0  [UNK]=1  [BOS]=2  [EOS]=3
  <|system|>=4   <|user|>=5   <|assistant|>=6   <|end|>=7

Usage
-----
    from tokenizer import MultilingualTokenizer

    tok = MultilingualTokenizer("tokenizer/multilingual.model")
    ids = tok.encode("Сәлем, қалайсыз?")
    text = tok.decode(ids)

    # Chat encoding
    ids = tok.encode_chat(
        system="You are a helpful assistant.",
        turns=[("user", "Hello!"), ("assistant", "Hi! How can I help?")]
    )
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional

import sentencepiece as spm


# ── Special tokens ─────────────────────────────────────────────────────────────

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]",
                  "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

PAD_ID       = 0
UNK_ID       = 1
BOS_ID       = 2
EOS_ID       = 3
SYSTEM_ID    = 4
USER_ID      = 5
ASSISTANT_ID = 6
END_ID       = 7

NUM_SPECIAL  = len(SPECIAL_TOKENS)   # 8


# ── Tokenizer class ────────────────────────────────────────────────────────────

class MultilingualTokenizer:
    """
    Thin wrapper around a SentencePiece BPE model with special chat tokens.

    The first NUM_SPECIAL (8) vocabulary slots are reserved for special tokens.
    SentencePiece piece IDs are shifted by +NUM_SPECIAL so they never collide.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                "Run  python tokenizer/train_tokenizer.py  first."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self._vocab_size = NUM_SPECIAL + self.sp.GetPieceSize()

    # ── Core encode / decode ───────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = [i + NUM_SPECIAL for i in self.sp.EncodeAsIds(text)]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Strip special tokens; shift back for SentencePiece
        sp_ids = [i - NUM_SPECIAL for i in ids if i >= NUM_SPECIAL]
        return self.sp.DecodeIds(sp_ids)

    # ── Chat encoding ──────────────────────────────────────────────────────────

    def encode_chat(
        self,
        turns: List[Tuple[str, str]],          # [("user", text), ("assistant", text), ...]
        system: Optional[str] = None,
        add_generation_prompt: bool = True,    # append <|assistant|> at end for inference
    ) -> List[int]:
        """
        Encode a conversation into token IDs.

        Format:
          [BOS] <|system|> {system_text} <|end|>
               <|user|> {user_text} <|end|>
               <|assistant|> {assistant_text} <|end|>
               ... [EOS]

        Returns a flat list of token IDs.
        """
        ids: List[int] = [BOS_ID]

        if system:
            ids += [SYSTEM_ID] + self.encode(system) + [END_ID]

        for role, text in turns:
            if role == "user":
                ids += [USER_ID] + self.encode(text) + [END_ID]
            elif role == "assistant":
                ids += [ASSISTANT_ID] + self.encode(text) + [END_ID]
            else:
                raise ValueError(f"Unknown role: {role}")

        if add_generation_prompt:
            ids += [ASSISTANT_ID]   # model continues from here
        else:
            ids += [EOS_ID]

        return ids

    def decode_chat_response(self, ids: List[int]) -> str:
        """
        Extract the last assistant turn from a generated token sequence.
        Stops at <|end|> or [EOS].
        """
        # Find last <|assistant|>
        start = 0
        for i, tok in enumerate(ids):
            if tok == ASSISTANT_ID:
                start = i + 1
        # Collect until <|end|> or [EOS]
        response_ids = []
        for tok in ids[start:]:
            if tok in (END_ID, EOS_ID):
                break
            response_ids.append(tok)
        return self.decode(response_ids)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    # ── Batch helpers ──────────────────────────────────────────────────────────

    def batch_encode(
        self,
        texts: List[str],
        max_len: int,
        pad: bool = True,
    ) -> "torch.Tensor":
        import torch
        encoded = [self.encode(t)[:max_len] for t in texts]
        if pad:
            max_l = max(len(e) for e in encoded)
            encoded = [e + [PAD_ID] * (max_l - len(e)) for e in encoded]
        return torch.tensor(encoded, dtype=torch.long)
