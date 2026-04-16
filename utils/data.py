"""
utils/data.py — shared data utilities for all three training stages.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Simple character-level tokeniser (swap for BPE in real use)
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Minimal character-level tokeniser.
    Suitable for demos and small experiments.
    Replace with tiktoken / SentencePiece for real training.
    """

    def __init__(self, text: Optional[str] = None, vocab_size: int = 256):
        if text is not None:
            chars = sorted(set(text))
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.itos = {i: c for c, i in self.stoi.items()}
            self.vocab_size = len(chars)
        else:
            # Fall back to raw byte encoding
            self.stoi = {chr(i): i for i in range(vocab_size)}
            self.itos = {i: chr(i) for i in range(vocab_size)}
            self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids)

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({'stoi': self.stoi}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        import json
        tok = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        tok.stoi = data['stoi']
        tok.itos = {v: k for k, v in tok.stoi.items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Flat token sequence split into (input, target) windows.

    input  : tokens[i : i+seq_len]
    target : tokens[i+1 : i+seq_len+1]
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 256):
        self.data    = token_ids
        self.seq_len = seq_len
        self.n       = len(token_ids) - seq_len - 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_text_dataset(
    path:    str,
    seq_len: int = 256,
    split:   float = 0.9,
) -> Tuple[TokenDataset, TokenDataset, CharTokenizer]:
    """
    Load a plain-text file and return (train_dataset, val_dataset, tokenizer).
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    tok  = CharTokenizer(text)
    ids  = torch.tensor(tok.encode(text), dtype=torch.long)
    n    = int(len(ids) * split)

    train_ds = TokenDataset(ids[:n],  seq_len)
    val_ds   = TokenDataset(ids[n:],  seq_len)
    return train_ds, val_ds, tok


def make_dummy_dataset(
    vocab_size: int = 256,
    n_tokens:   int = 50_000,
    seq_len:    int = 256,
) -> TokenDataset:
    """Random token dataset for smoke-testing."""
    data = torch.randint(0, vocab_size, (n_tokens,))
    return TokenDataset(data, seq_len)


def get_dataloader(
    dataset:    Dataset,
    batch_size: int  = 1,
    shuffle:    bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Random batch generator (for unit tests / demos without a Dataset object)
# ──────────────────────────────────────────────────────────────────────────────

def random_batch(
    vocab_size: int,
    seq_len:    int,
    batch_size: int,
    device:     torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, target_ids) as random longs on device."""
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(path: str, device: torch.device = torch.device('cpu')) -> dict:
    ckpt = torch.load(path, map_location=device)
    print(f"[ckpt] Loaded ← {path}")
    return ckpt
