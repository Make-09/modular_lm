"""
Block 1 — Syntax Engine
=======================
Purpose : Learn token structure, grammar, positional ordering.
          NOT optimised for factual knowledge (reduced FFN multiplier = 2).

Key design choices
------------------
* Standard causal (decoder-style) transformer with weight-tied embedding/LM head.
* FFN multiplier = 2  (vs. 4 in GPT-2) to limit fact-memorisation capacity.
* Gradient checkpointing support for Stage-1 training under 4 GB VRAM.
* `forward()` returns z_struct  (B, T, d_model) for downstream blocks.
* `forward_lm()` additionally returns next-token logits for Stage-1 LM loss.

Approximate parameter count (defaults)
---------------------------------------
  token_emb  : vocab_size × d_model    ≈ 16.4 M   (shared with lm_head)
  pos_emb    : max_seq_len × d_model   ≈  0.1 M
  6 × layer  :
      self-attn (QKV + out) : 4 × d²   ≈  1.05 M
      FFN (d→2d→d)          :           ≈  0.52 M
      layer-norms           :           ≈     ~0
  Total (shared lm_head)               ≈ 29 M
"""
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        x           : (B, T, d_model)
        causal_mask : (1, 1, T, T)  lower-triangular boolean
        Returns     : (B, T, d_model)
        """
        B, T, C = x.shape

        # ── Project and split heads ──
        qkv = self.qkv(x)                                         # (B, T, 3C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, H, T, dh)
        q, k, v = qkv.unbind(0)                                   # each (B, H, T, dh)

        # ── Scaled dot-product attention ──
        attn = (q @ k.transpose(-2, -1)) / self.scale             # (B, H, T, T)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                             # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, C)                # (B, T, C)
        return self.out_proj(out)


class ReducedFFN(nn.Module):
    """
    Feed-forward network with ffn_mult=2 (half of GPT-2's 4×).
    Smaller capacity → less fact memorisation, more syntax focus.
    """

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


class SyntaxLayer(nn.Module):
    """Single transformer layer: pre-norm → self-attn → pre-norm → FFN."""

    def __init__(self, d_model: int, n_heads: int,
                 ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = ReducedFFN(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Block 1
# ──────────────────────────────────────────────────────────────────────────────

class Block1SyntaxEngine(nn.Module):
    """
    Syntax Engine — the structural backbone of the modular LM.

    Tensor shapes (with defaults d_model=512, seq_len=T):
      input  : (B, T)  long  — token ids
      output : (B, T, 512)   — z_struct, contextual embeddings
      logits : (B, T, vocab_size) — only when forward_lm() is called
    """

    def __init__(
        self,
        vocab_size:  int   = 32_000,
        d_model:     int   = 512,
        n_heads:     int   = 8,
        n_layers:    int   = 6,
        max_seq_len: int   = 256,
        ffn_mult:    int   = 2,
        dropout:     float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.use_ckpt   = use_gradient_checkpointing

        # ── Embeddings ──
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Transformer layers ──
        self.layers = nn.ModuleList([
            SyntaxLayer(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── LM head (weight-tied to token_emb) ──
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Scale embedding init
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular mask: (1, 1, T, T) of 0/1."""
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    # ── Forward passes ──────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Primary forward pass — used by Blocks 2 & 3 downstream.

        input_ids : (B, T)
        Returns   : z_struct  (B, T, d_model)
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos  = torch.arange(T, device=device).unsqueeze(0)            # (1, T)
        x    = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = self._causal_mask(T, device)

        for layer in self.layers:
            if self.use_ckpt and self.training:
                # Wrap layer call for gradient checkpointing
                x = ckpt.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

        return self.norm(x)   # z_struct

    def forward_lm(self, input_ids: torch.Tensor):
        """
        Stage-1 training pass — returns (logits, z_struct).

        logits   : (B, T, vocab_size)
        z_struct : (B, T, d_model)
        """
        z_struct = self.forward(input_ids)
        logits   = self.lm_head(z_struct)
        return logits, z_struct

    # ── Utility ─────────────────────────────────────────────────────────────

    def freeze(self):
        """Freeze all parameters after Stage 1 training."""
        for p in self.parameters():
            p.requires_grad = False
        print("[Block1] Weights frozen.")

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        fn = (lambda p: p.requires_grad) if trainable_only else (lambda p: True)
        return sum(p.numel() for p in self.parameters() if fn(p))

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters())
        return (f"Block1SyntaxEngine("
                f"d_model={self.d_model}, layers={self.n_layers}, "
                f"params={n/1e6:.2f}M)")
