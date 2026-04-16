"""
Block 2 — Knowledge Memory (Product-Key Memory)
================================================
Purpose : Store facts, entities, and associations that Block 1 deliberately
          avoids memorising.

Architecture : Product-Key Memory (PKM)
-----------------------------------------
PKM factorises a large key space into two independent sub-key sets so that
|K|² memory slots can be addressed using only 2|K| parameters.

Reference : Lample et al. (2019) "Large Memory Layers with Product Keys"
            https://arxiv.org/abs/1907.05242

Data flow
---------
  query q  (B, T, d_model)
      │
      ▼ query_proj → (B, T, d_memory)
      │             reshape → (B·T, n_heads, d_head)
      │
      ├─ dot-product with keys_left  → scores_left  (BT, H, num_keys)
      ├─ dot-product with keys_right → scores_right (BT, H, num_keys)
      │
      ▼ top-k from each half, outer-product → top_k² candidate pairs
        softmax over product scores
        weighted sum of values[pairs]
      │
      ▼ out_proj → z_memory (B, T, d_model)

Parameter budget (defaults: num_keys=128, top_k=16, n_heads=4, d_memory=256)
-----------------------------------------------------------------------------
  query_proj  : d_model × d_memory        =  0.13 M
  keys_left   : n_heads × num_keys × dh   =  32 K
  keys_right  : n_heads × num_keys × dh   =  32 K
  values      : n_heads × num_keys² × dh  =  4.2 M   (the "knowledge store")
  out_proj    : d_memory × d_model         =  0.13 M
  ──────────────────────────────────────────────────
  Total                                   ≈  4.5 M

  → updatable in-place via update_memory() without touching Block 1 or 3.

Shapes legend
-------------
  B  = batch size
  T  = sequence length
  H  = num_heads
  K  = num_keys  (sub-key count per half)
  dh = d_memory // num_heads  (per-head dimension)
  k  = top_k
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Block 2
# ──────────────────────────────────────────────────────────────────────────────

class Block2ProductKeyMemory(nn.Module):
    """
    Product-Key Memory module.

    Key properties
    --------------
    * CPU-friendly inference  : values are an nn.Embedding table; lookup is O(H·k²).
    * Hot-swappable knowledge : call update_memory() to replace stored facts.
    * Multi-head retrieval    : H parallel memory channels per token.
    """

    def __init__(
        self,
        d_model:   int   = 512,
        d_memory:  int   = 256,    # internal memory width
        num_keys:  int   = 128,    # sub-key count per half  (total slots = num_keys²)
        top_k:     int   = 16,     # top-k per sub-key set
        num_heads: int   = 4,
        dropout:   float = 0.0,
    ):
        super().__init__()
        assert d_memory % num_heads == 0, "d_memory must be divisible by num_heads"

        self.d_model   = d_model
        self.d_memory  = d_memory
        self.num_keys  = num_keys
        self.top_k     = top_k
        self.num_heads = num_heads
        self.d_head    = d_memory // num_heads   # dh

        # ── Query projection (Block-1 output → memory query) ──
        self.query_proj = nn.Linear(d_model, d_memory, bias=False)

        # ── Sub-key matrices  (H, K, dh) ──
        self.keys_left  = nn.Parameter(
            torch.empty(num_heads, num_keys, self.d_head)
        )
        self.keys_right = nn.Parameter(
            torch.empty(num_heads, num_keys, self.d_head)
        )

        # ── Value store  Embedding(H·K², dh) ──
        # Each head has num_keys² slots; slot h*K²+i*K+j stores value for
        # the pair (left_key i, right_key j) under head h.
        n_slots = num_heads * num_keys * num_keys
        self.values = nn.Embedding(n_slots, self.d_head)

        # ── Output projection ──
        self.out_proj = nn.Linear(d_memory, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        self._init_weights()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.keys_left,  std=1.0 / math.sqrt(self.d_head))
        nn.init.normal_(self.keys_right, std=1.0 / math.sqrt(self.d_head))
        nn.init.normal_(self.values.weight, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    # ── Product-key retrieval ────────────────────────────────────────────────

    def _product_retrieve(
        self,
        scores_left:  torch.Tensor,   # (BT, H, K)
        scores_right: torch.Tensor,   # (BT, H, K)
        device:       torch.device,
    ):
        """
        1. Top-k from each half.
        2. Outer-product of scores → top_k² candidates.
        3. Return flat slot indices + their combined scores.

        Returns
        -------
        slot_indices : (BT, H, k²)   — absolute value-table indices
        prod_scores  : (BT, H, k²)
        """
        k = self.top_k

        # Top-k from each half  →  (BT, H, k)
        left_scores,  left_idx  = scores_left.topk( k, dim=-1)
        right_scores, right_idx = scores_right.topk(k, dim=-1)

        # Outer-product of scores  →  (BT, H, k, k)
        prod_scores = left_scores.unsqueeze(-1) + right_scores.unsqueeze(-2)

        # Pair indices: left * K + right  →  (BT, H, k, k)
        pair_idx = (
            left_idx.unsqueeze(-1) * self.num_keys
            + right_idx.unsqueeze(-2)
        )   # relative within-head slot index

        # Flatten k×k  →  (BT, H, k²)
        k2 = k * k
        prod_scores = prod_scores.reshape(*prod_scores.shape[:-2], k2)
        pair_idx    = pair_idx.reshape(*pair_idx.shape[:-2], k2)

        # Add per-head offset so we can index into the flat Embedding table
        # head h occupies rows [h·K², (h+1)·K²)
        head_offsets = (
            torch.arange(self.num_heads, device=device) * (self.num_keys ** 2)
        ).view(1, self.num_heads, 1)                              # (1, H, 1)
        slot_indices = pair_idx + head_offsets                    # (BT, H, k²)

        return slot_indices, prod_scores

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        query    : (B, T, d_model)  — projected Block-1 output (q_memory)
        Returns  : z_memory  (B, T, d_model)
        """
        B, T, _ = query.shape
        BT = B * T
        device = query.device

        # ── 1. Project query  →  (BT, H, dh) ──
        q = self.query_proj(query)                                 # (B, T, d_memory)
        q = q.reshape(BT, self.num_heads, self.d_head)

        # ── 2. L2-normalise for cosine similarity ──
        q_n  = F.normalize(q,              dim=-1)                 # (BT, H, dh)
        kl_n = F.normalize(self.keys_left, dim=-1)                 # (H, K, dh)
        kr_n = F.normalize(self.keys_right, dim=-1)

        # ── 3. Dot-products with each sub-key set ──
        # einsum: (BT, H, dh) · (H, K, dh)^T → (BT, H, K)
        scores_left  = torch.einsum('bhd,hkd->bhk', q_n, kl_n)
        scores_right = torch.einsum('bhd,hkd->bhk', q_n, kr_n)

        # ── 4. Product-key retrieval ──
        slot_idx, prod_scores = self._product_retrieve(
            scores_left, scores_right, device
        )
        # slot_idx : (BT, H, k²)   int64
        # prod_scores: (BT, H, k²) float

        # ── 5. Softmax attention over k² candidates ──
        attn_w = F.softmax(prod_scores, dim=-1)                    # (BT, H, k²)
        attn_w = self.dropout(attn_w)

        # ── 6. Retrieve values and aggregate ──
        flat_idx  = slot_idx.reshape(-1)                           # (BT·H·k²,)
        retrieved = self.values(flat_idx)                          # (BT·H·k², dh)
        retrieved = retrieved.reshape(BT, self.num_heads,
                                      self.top_k ** 2, self.d_head)

        # Weighted sum over k² candidates
        z = (attn_w.unsqueeze(-1) * retrieved).sum(dim=2)         # (BT, H, dh)

        # ── 7. Concat heads, project out ──
        z = z.reshape(B, T, self.d_memory)                        # (B, T, d_memory)
        z_memory = self.out_proj(z)                                # (B, T, d_model)
        z_memory = self.norm_out(z_memory)

        return z_memory

    # ── External memory management ──────────────────────────────────────────

    def update_memory(
        self,
        new_keys_left:  torch.Tensor | None = None,
        new_keys_right: torch.Tensor | None = None,
        new_values:     torch.Tensor | None = None,
    ):
        """
        Hot-swap stored knowledge without retraining.

        Parameters
        ----------
        new_keys_left  : (num_heads, num_keys, d_head)
        new_keys_right : (num_heads, num_keys, d_head)
        new_values     : (num_heads * num_keys², d_head)
        """
        with torch.no_grad():
            if new_keys_left is not None:
                self.keys_left.copy_(new_keys_left.to(self.keys_left.device))
            if new_keys_right is not None:
                self.keys_right.copy_(new_keys_right.to(self.keys_right.device))
            if new_values is not None:
                self.values.weight.copy_(new_values.to(self.values.weight.device))
        print("[Block2] Memory updated in-place.")

    def export_memory(self) -> dict:
        """Export current memory state (for serialisation / migration)."""
        return {
            'keys_left':  self.keys_left.detach().cpu(),
            'keys_right': self.keys_right.detach().cpu(),
            'values':     self.values.weight.detach().cpu(),
        }

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[Block2] Memory frozen.")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        fn = (lambda p: p.requires_grad) if trainable_only else (lambda p: True)
        return sum(p.numel() for p in self.parameters() if fn(p))

    def __repr__(self):
        n      = sum(p.numel() for p in self.parameters())
        slots  = self.num_heads * self.num_keys ** 2
        return (f"Block2ProductKeyMemory("
                f"num_heads={self.num_heads}, num_keys={self.num_keys}, "
                f"top_k={self.top_k}, slots={slots:,}, params={n/1e6:.2f}M)")
