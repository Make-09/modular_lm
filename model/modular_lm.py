"""
ModularLM — Full Architecture
==============================

Full data flow
--------------

  input_ids  (B, T)
       │
       ▼ ──────────────────────────────────────────────────────────────────────
       │                      BLOCK 1  (frozen after Stage 1)
       │  token_emb + pos_emb → 6× SyntaxLayer (causal self-attn + reduced FFN)
       ▼
  z_struct   (B, T, 512)         ← contextual syntax embeddings
       │
       ├────────────────────────────────────────────────────────────────────────
       │                   MEMORY QUERY PROJECTION
       │  Linear(512→512) + GELU + Linear(512→512) + LayerNorm
       ▼
  q_memory   (B, T, 512)         ← query into knowledge store
       │
       ▼ ──────────────────────────────────────────────────────────────────────
       │                      BLOCK 2  (frozen after Stage 2)
       │  Product-Key Memory: top-k² retrieval across H·K² slots
       ▼
  z_memory   (B, T, 512)         ← retrieved factual vectors
       │
       └─────────────────────────┐
                                 ▼ ─────────────────────────────────────────────
                            BLOCK 3  (trained in Stage 3)
                     z_struct ─→ cross-attn-struct ┐
                     z_memory ─→ cross-attn-memory ├─ 4× FusionLayer
                                 self-attn + FFN   ┘
                                 ▼
                            logits  (B, T, vocab_size)

Auxiliary loss
--------------
  L_total = L_lm  +  λ · L_memory_entropy
  L_memory_entropy  penalises low-entropy memory attention weights.
  This ensures Block 3 actively reads from z_memory.

Parameter budget (defaults)
----------------------------
  Block 1 (Syntax)    : ~29 M
  Memory projection   :  0.5 M
  Block 2 (Memory)    :  4.5 M
  Block 3 (Fusion)    : ~15 M   (lm_head shared with Block 1's embedding)
  ─────────────────────────────
  Total               : ~49 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .block1_syntax import Block1SyntaxEngine
from .block2_memory import Block2ProductKeyMemory
from .block3_fusion import Block3FusionModule


# ──────────────────────────────────────────────────────────────────────────────
# Block 1 → Block 2 interface
# ──────────────────────────────────────────────────────────────────────────────

class MemoryQueryProjection(nn.Module):
    """
    Learnable projection from z_struct → q_memory.

    Two-layer MLP with GELU + LayerNorm. Trained in Stage 2 alongside Block 2,
    optionally fine-tuned in Stage 3.

    Shapes : (B, T, d_model) → (B, T, d_model)
    Params : d_model² × 2 ≈ 0.5 M  (d_model=512)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z_struct: torch.Tensor) -> torch.Tensor:
        return self.proj(z_struct)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class ModularLM(nn.Module):
    """
    Three-block modular language model.

    Constructor args
    ----------------
    All block configs are passed flat; see defaults below.
    Use `from_config(cfg_dict)` for dict-based construction.
    """

    def __init__(
        self,
        # ── Shared ──
        vocab_size:  int   = 32_000,
        d_model:     int   = 512,
        max_seq_len: int   = 256,
        dropout:     float = 0.1,
        # ── Block 1 ──
        b1_n_heads:  int   = 8,
        b1_n_layers: int   = 6,
        b1_ffn_mult: int   = 2,
        # ── Block 2 ──
        b2_d_memory:  int  = 256,
        b2_num_keys:  int  = 128,
        b2_top_k:     int  = 16,
        b2_num_heads: int  = 4,
        # ── Block 3 ──
        b3_n_heads:       int   = 8,
        b3_n_layers:      int   = 4,
        b3_ffn_mult:      int   = 2,
        b3_struct_dropout:float = 0.2,
        # ── Misc ──
        use_gradient_checkpointing: bool = True,
        # ── Aux loss weight ──
        memory_aux_lambda:   float = 0.1,
        memory_target_entropy: float = 2.0,
    ):
        super().__init__()

        self.memory_aux_lambda    = memory_aux_lambda
        self.memory_target_entropy = memory_target_entropy

        # ── Block 1 ──────────────────────────────────────────────────────────
        self.block1 = Block1SyntaxEngine(
            vocab_size  = vocab_size,
            d_model     = d_model,
            n_heads     = b1_n_heads,
            n_layers    = b1_n_layers,
            max_seq_len = max_seq_len,
            ffn_mult    = b1_ffn_mult,
            dropout     = dropout,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )

        # ── Interface Block 1 → Block 2 ──────────────────────────────────────
        self.memory_proj = MemoryQueryProjection(d_model, dropout)

        # ── Block 2 ──────────────────────────────────────────────────────────
        self.block2 = Block2ProductKeyMemory(
            d_model   = d_model,
            d_memory  = b2_d_memory,
            num_keys  = b2_num_keys,
            top_k     = b2_top_k,
            num_heads = b2_num_heads,
        )

        # ── Block 3 ──────────────────────────────────────────────────────────
        # Share the LM head with Block 1's embedding (weight tying across blocks)
        self.block3 = Block3FusionModule(
            vocab_size   = vocab_size,
            d_model      = d_model,
            n_heads      = b3_n_heads,
            n_layers     = b3_n_layers,
            max_seq_len  = max_seq_len,
            ffn_mult     = b3_ffn_mult,
            dropout      = dropout,
            struct_dropout = b3_struct_dropout,
            use_gradient_checkpointing = use_gradient_checkpointing,
            shared_lm_head = self.block1.lm_head,   # ← weight sharing
        )

    # ── Stage-management helpers ─────────────────────────────────────────────

    def prepare_stage1(self):
        """Stage 1: only Block 1 is trainable."""
        self._set_requires_grad(self.block1,      True)
        self._set_requires_grad(self.memory_proj, False)
        self._set_requires_grad(self.block2,      False)
        self._set_requires_grad(self.block3,      False)
        print("[ModularLM] Stage 1 ready — Block 1 trainable.")

    def prepare_stage2(self):
        """Stage 2: Block 1 frozen; memory_proj + Block 2 trainable."""
        self._set_requires_grad(self.block1,      False)
        self._set_requires_grad(self.memory_proj, True)
        self._set_requires_grad(self.block2,      True)
        self._set_requires_grad(self.block3,      False)
        print("[ModularLM] Stage 2 ready — Block 2 + projection trainable.")

    def prepare_stage3(self):
        """Stage 3: Block 1 & 2 frozen; Block 3 trainable."""
        self._set_requires_grad(self.block1,      False)
        self._set_requires_grad(self.memory_proj, False)
        self._set_requires_grad(self.block2,      False)
        self._set_requires_grad(self.block3,      True)
        print("[ModularLM] Stage 3 ready — Block 3 trainable.")

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    # ── Forward passes ──────────────────────────────────────────────────────

    def forward_stage1(self, input_ids: torch.Tensor):
        """
        Stage-1 training forward pass (Block 1 only).

        Returns
        -------
        logits   : (B, T, vocab_size)
        z_struct : (B, T, d_model)
        """
        return self.block1.forward_lm(input_ids)

    def forward_stage2(self, input_ids: torch.Tensor):
        """
        Stage-2 training forward pass (memory query + Block 2).
        Block 1 is called in no_grad mode.

        Returns
        -------
        z_memory : (B, T, d_model)
        q_memory : (B, T, d_model)   — for optional regularisation
        """
        with torch.no_grad():
            z_struct = self.block1(input_ids)

        q_memory = self.memory_proj(z_struct)
        z_memory = self.block2(q_memory)
        return z_memory, q_memory

    def forward(
        self,
        input_ids: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Full forward pass (Stages 2–3 combined / inference).

        Returns
        -------
        logits   : (B, T, vocab_size)
        aux_info : dict  (None if return_aux=False)
        """
        # ── Block 1 (no grad if frozen) ──
        z_struct = self.block1(input_ids)                         # (B, T, d_model)

        # ── Block 1 → Block 2 interface ──
        q_memory = self.memory_proj(z_struct)                     # (B, T, d_model)

        # ── Block 2 ──
        z_memory = self.block2(q_memory)                          # (B, T, d_model)

        # ── Block 3 ──
        logits, aux_info = self.block3(z_struct, z_memory)        # (B, T, vocab_size)

        if not return_aux:
            return logits, None
        return logits, aux_info

    # ── Auxiliary loss ───────────────────────────────────────────────────────

    def memory_entropy_loss(self, aux_info: Dict) -> torch.Tensor:
        """
        Anti-collapse auxiliary loss.

        Computes the mean attention entropy for the memory cross-attention
        across all layers. Penalises low entropy (= model ignoring memory).

        L_mem = mean_over_layers(  relu(H_target − H_actual)  )

        where H = −Σ p·log(p) is Shannon entropy of the attention distribution.
        """
        losses = []
        for attn_m in aux_info['attn_memory']:
            # attn_m : (B, n_heads, T, T)
            p = attn_m.clamp(min=1e-9)
            H = -(p * p.log()).sum(dim=-1).mean()   # scalar
            # Penalise if entropy is below target
            losses.append(F.relu(self.memory_target_entropy - H))

        return sum(losses) / max(len(losses), 1)

    def loss(
        self,
        logits:   torch.Tensor,
        targets:  torch.Tensor,
        aux_info: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Combined next-token + auxiliary memory-entropy loss.

        Returns
        -------
        total_loss  : scalar
        loss_dict   : {'lm': ..., 'memory_entropy': ..., 'total': ...}
        """
        B, T, V = logits.shape
        lm_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
        )

        if aux_info is not None and self.training:
            mem_loss  = self.memory_entropy_loss(aux_info)
            total     = lm_loss + self.memory_aux_lambda * mem_loss
        else:
            mem_loss  = torch.tensor(0.0, device=logits.device)
            total     = lm_loss

        return total, {
            'lm':              lm_loss.item(),
            'memory_entropy':  mem_loss.item(),
            'total':           total.item(),
        }

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def count_parameters(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())

        b1 = n(self.block1)
        mp = n(self.memory_proj)
        b2 = n(self.block2)
        b3 = n(self.block3)

        print("=" * 50)
        print(f"  Block 1  (Syntax)     : {b1/1e6:7.2f} M")
        print(f"  Memory Projection     : {mp/1e6:7.2f} M")
        print(f"  Block 2  (Memory)     : {b2/1e6:7.2f} M")
        print(f"  Block 3  (Fusion)     : {b3/1e6:7.2f} M")
        print(f"  ─ shared lm_head not double-counted ─")
        # lm_head weight is shared — count it once
        lm_head_params = self.block1.lm_head.weight.numel()
        unique_total   = b1 + mp + b2 + b3 - lm_head_params
        print(f"  Total (unique)        : {unique_total/1e6:7.2f} M")
        print("=" * 50)
        return unique_total

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict) -> "ModularLM":
        return cls(**cfg)
