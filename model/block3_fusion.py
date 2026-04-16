"""
Block 3 — Fusion / Working Memory
===================================
Purpose : Combine structural information (z_struct from Block 1) with retrieved
          factual knowledge (z_memory from Block 2) and produce final token
          predictions.

Architecture
------------
Each FusionLayer contains, in order:
  1. Causal self-attention          (working-memory dynamics)
  2. Cross-attention → z_struct     (syntax context)
  3. Cross-attention → z_memory     (factual context)
  4. Small FFN (ffn_mult=2)         (composition / bottleneck)

Anti-collapse mechanisms
------------------------
* Dropout on z_struct input during training  (struct_dropout=0.2)
  Forces the fusion block to learn from memory even when syntax is noisy.
* Bottleneck FFN  (mult=2, not 4)
  Limits the capacity to "re-derive" facts from structural patterns alone.
* Auxiliary memory-attention loss (computed externally, see modular_lm.py)
  Penalises low-entropy memory attention, preventing the model from
  effectively zeroing out z_memory cross-attention.
* SEPARATE cross-attention modules for z_struct and z_memory — they cannot
  trivially be fused/ignored by a single gate.

Data flow per layer
-------------------
  x           (B, T, d_model)  ← fusion hidden state
  z_struct_d  (B, T, d_model)  ← Block-1 output (with dropout)
  z_memory    (B, T, d_model)  ← Block-2 output

  x = x + CausalSelfAttn(LN(x))
  x = x + CrossAttnStruct(LN(x), z_struct_d)     [attn_struct returned for aux loss]
  x = x + CrossAttnMemory(LN(x), z_memory)        [attn_memory returned for aux loss]
  x = x + FFN(LN(x))

Parameter count (defaults: d_model=512, n_heads=8, n_layers=4, ffn_mult=2)
---------------------------------------------------------------------------
  4 × FusionLayer:
      self-attn   : 4 × 512²  ≈ 1.05 M
      xattn-struct: 4 × 512²  ≈ 1.05 M
      xattn-mem   : 4 × 512²  ≈ 1.05 M
      FFN         :            ≈ 0.52 M
                              ≈ 3.67 M / layer
  4 layers                   ≈ 14.7 M
  (lm_head shared with Block 1 token_emb — 0 extra params)
  Total                      ≈ 15 M
"""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from typing import List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention (standard decoder style)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: queries from fusion state, keys/values from external source.
    Returns both the output and the raw (pre-dropout) attention weights for the
    auxiliary memory-utilisation loss.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        query:   torch.Tensor,    # (B, T,  d_model)  — fusion hidden state
        context: torch.Tensor,    # (B, T', d_model)  — z_struct or z_memory
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out          : (B, T, d_model)
        attn_weights : (B, n_heads, T, T')   — for auxiliary loss
        """
        B, T,  _ = query.shape
        _,  Tc, _ = context.shape

        def split_heads(t, length):
            return t.reshape(B, length, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(self.q_proj(query),   T)                   # (B, H, T, dh)
        k = split_heads(self.k_proj(context), Tc)
        v = split_heads(self.v_proj(context), Tc)

        attn_raw = (q @ k.transpose(-2, -1)) / self.scale          # (B, H, T, T')
        attn_w   = torch.softmax(attn_raw, dim=-1)                 # saved for aux loss
        attn_d   = self.drop(attn_w)

        out = (attn_d @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(out), attn_w


class FusionFFN(nn.Module):
    """Small bottleneck FFN (mult=2) to limit re-derivation of facts."""

    def __init__(self, d_model: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * ffn_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Single fusion layer
# ──────────────────────────────────────────────────────────────────────────────

class FusionLayer(nn.Module):
    """
    One layer of the fusion / working-memory block.

    Enforces use of BOTH inputs through separate cross-attention modules
    that cannot be trivially bypassed.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ffn_mult: int   = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()

        # ── 1. Causal self-attention ──
        self.norm_self   = nn.LayerNorm(d_model)
        self.self_attn   = MultiHeadSelfAttention(d_model, n_heads, dropout)

        # ── 2. Cross-attention → z_struct ──
        self.norm_struct = nn.LayerNorm(d_model)
        self.xattn_struct = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # ── 3. Cross-attention → z_memory ──
        self.norm_mem    = nn.LayerNorm(d_model)
        self.xattn_mem   = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # ── 4. FFN ──
        self.norm_ffn    = nn.LayerNorm(d_model)
        self.ffn         = FusionFFN(d_model, ffn_mult, dropout)

    def forward(
        self,
        x:        torch.Tensor,   # (B, T, d_model)
        z_struct: torch.Tensor,   # (B, T, d_model)
        z_memory: torch.Tensor,   # (B, T, d_model)
        causal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x            : (B, T, d_model)  updated fusion state
        attn_struct  : (B, n_heads, T, T)
        attn_memory  : (B, n_heads, T, T)
        """
        # 1. Causal self-attention (own context)
        x = x + self.self_attn(self.norm_self(x), causal_mask)

        # 2. Syntax input — cross-attend to z_struct
        struct_delta, attn_struct = self.xattn_struct(self.norm_struct(x), z_struct)
        x = x + struct_delta

        # 3. Knowledge input — cross-attend to z_memory
        mem_delta, attn_memory = self.xattn_mem(self.norm_mem(x), z_memory)
        x = x + mem_delta

        # 4. Bottleneck FFN
        x = x + self.ffn(self.norm_ffn(x))

        return x, attn_struct, attn_memory


# ──────────────────────────────────────────────────────────────────────────────
# Block 3
# ──────────────────────────────────────────────────────────────────────────────

class Block3FusionModule(nn.Module):
    """
    Fusion / Working Memory block.

    Inputs (both required)
    ----------------------
    z_struct  (B, T, d_model) — frozen Block-1 contextual embeddings
    z_memory  (B, T, d_model) — Block-2 retrieved knowledge

    Output
    ------
    logits    (B, T, vocab_size)
    aux_info  dict with attention tensors for auxiliary loss computation

    Anti-collapse design
    --------------------
    * struct_dropout applied to z_struct during training
    * Separate cross-attention modules — model cannot short-circuit one stream
    * Bottleneck FFN prevents re-learning facts from scratch
    * attn_memory weights exposed for entropy-based auxiliary loss
    """

    def __init__(
        self,
        vocab_size:      int   = 32_000,
        d_model:         int   = 512,
        n_heads:         int   = 8,
        n_layers:        int   = 4,
        max_seq_len:     int   = 256,
        ffn_mult:        int   = 2,
        dropout:         float = 0.1,
        struct_dropout:  float = 0.2,
        use_gradient_checkpointing: bool = True,
        # LM head: can optionally share with Block-1 embedding (passed externally)
        shared_lm_head: nn.Linear | None = None,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.use_ckpt   = use_gradient_checkpointing

        # Anti-collapse: dropout on z_struct input
        self.struct_drop = nn.Dropout(struct_dropout)

        # Fusion layers
        self.layers = nn.ModuleList([
            FusionLayer(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # LM head — share with Block 1's token embedding if provided
        if shared_lm_head is not None:
            self.lm_head = shared_lm_head
            self._owns_lm_head = False
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self._owns_lm_head = True

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'lm_head' in name and not self._owns_lm_head:
                continue  # shared — don't reinitialise
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    # ── Core checkpoint-compatible layer runner ──────────────────────────────

    def _run_layer(
        self,
        layer: FusionLayer,
        x:     torch.Tensor,
        zs:    torch.Tensor,
        zm:    torch.Tensor,
        mask:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Thin wrapper so gradient checkpointing can wrap the layer call."""
        return layer(x, zs, zm, mask)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        z_struct:  torch.Tensor,    # (B, T, d_model)  — from frozen Block 1
        z_memory:  torch.Tensor,    # (B, T, d_model)  — from Block 2
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        logits   : (B, T, vocab_size)
        aux_info : {
            'attn_struct' : List[(B, n_heads, T, T)]   one per layer
            'attn_memory' : List[(B, n_heads, T, T)]   one per layer
          }
        """
        B, T, _ = z_struct.shape
        device   = z_struct.device

        # Anti-collapse: stochastic dropout on syntax signal
        z_struct_d = self.struct_drop(z_struct) if self.training else z_struct

        # Initialise fusion state from (dropped) syntax embeddings
        x    = z_struct_d.clone()
        mask = self._causal_mask(T, device)

        all_attn_struct: List[torch.Tensor] = []
        all_attn_memory: List[torch.Tensor] = []

        for layer in self.layers:
            if self.use_ckpt and self.training:
                # gradient checkpointing trades compute for memory
                def make_fn(ly):
                    def fn(x_, zs_, zm_, m_):
                        return ly(x_, zs_, zm_, m_)
                    return fn
                # ckpt does not support multiple returns directly → wrap
                x, attn_s, attn_m = ckpt.checkpoint(
                    make_fn(layer), x, z_struct_d, z_memory, mask,
                    use_reentrant=False
                )
            else:
                x, attn_s, attn_m = layer(x, z_struct_d, z_memory, mask)

            all_attn_struct.append(attn_s)
            all_attn_memory.append(attn_m)

        x      = self.norm(x)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)

        aux_info = {
            'attn_struct': all_attn_struct,
            'attn_memory': all_attn_memory,
        }
        return logits, aux_info

    # ── Utility ─────────────────────────────────────────────────────────────

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[Block3] Weights frozen.")

    def count_parameters(self, trainable_only: bool = True) -> int:
        fn = (lambda p: p.requires_grad) if trainable_only else (lambda p: True)
        return sum(p.numel() for p in self.parameters() if fn(p))

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters())
        return (f"Block3FusionModule("
                f"d_model={self.d_model}, layers={self.n_layers}, "
                f"params={n/1e6:.2f}M)")
