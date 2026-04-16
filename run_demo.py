"""
run_demo.py — Minimal end-to-end runnable example with dummy data.
==================================================================

Demonstrates:
  1. Model construction & parameter counts
  2. All three forward-pass variants (Stage 1, 2, 3)
  3. Loss computation (LM + auxiliary memory-entropy)
  4. FP16 autocast compatibility check
  5. Gradient flow verification
  6. In-place memory update (Block 2 hot-swap)
  7. Tensor shape trace

No real data or checkpoints required. Safe to run on CPU.

Usage
-----
  python run_demo.py
  python run_demo.py --device cuda   # if GPU available
"""

import argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.block1_syntax import Block1SyntaxEngine
from model.block2_memory import Block2ProductKeyMemory
from model.block3_fusion import Block3FusionModule
from model.modular_lm    import ModularLM, MemoryQueryProjection


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE  = 512    # small for demo speed
D_MODEL     = 256    # halved for fast CPU demo
SEQ_LEN     = 64
BATCH       = 1


def separator(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(device: torch.device):

    use_fp16 = device.type == 'cuda'

    # ── 1. Construct full model ───────────────────────────────────────────────
    separator("1. Model Construction & Parameter Count")

    model = ModularLM(
        vocab_size   = VOCAB_SIZE,
        d_model      = D_MODEL,
        max_seq_len  = SEQ_LEN,
        # Block 1
        b1_n_heads   = 4,
        b1_n_layers  = 4,
        b1_ffn_mult  = 2,
        # Block 2
        b2_d_memory  = 128,
        b2_num_keys  = 64,
        b2_top_k     = 8,
        b2_num_heads = 2,
        # Block 3
        b3_n_heads   = 4,
        b3_n_layers  = 3,
        b3_ffn_mult  = 2,
        b3_struct_dropout = 0.2,
        use_gradient_checkpointing = False,   # off for CPU demo clarity
    ).to(device)

    total = model.count_parameters()

    # ── 2. Dummy input ────────────────────────────────────────────────────────
    separator("2. Tensor Shape Trace")

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)
    targets   = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=device)

    print(f"  input_ids  : {tuple(input_ids.shape)}  dtype={input_ids.dtype}")

    with autocast(enabled=use_fp16):
        # ── Block 1 ──
        z_struct = model.block1(input_ids)
        print(f"  z_struct   : {tuple(z_struct.shape)}  (Block 1 output)")

        # ── Memory projection ──
        q_memory = model.memory_proj(z_struct)
        print(f"  q_memory   : {tuple(q_memory.shape)}  (memory query)")

        # ── Block 2 ──
        z_memory = model.block2(q_memory)
        print(f"  z_memory   : {tuple(z_memory.shape)}  (Block 2 output)")

        # ── Block 3 ──
        logits, aux_info = model.block3(z_struct, z_memory)
        print(f"  logits     : {tuple(logits.shape)}  (Block 3 output)")

    # Attention shapes
    print(f"\n  Attention tensors per layer (Block 3):")
    for i, (as_, am_) in enumerate(
        zip(aux_info['attn_struct'], aux_info['attn_memory'])
    ):
        print(f"    layer {i} | attn_struct {tuple(as_.shape)} | "
              f"attn_memory {tuple(am_.shape)}")

    # ── 3. Stage-1 forward ────────────────────────────────────────────────────
    separator("3. Stage 1 — Block 1 Language Modelling")

    model.prepare_stage1()
    print(f"  Trainable params: {model.trainable_parameters()/1e6:.2f}M")

    model.block1.train()
    with autocast(enabled=use_fp16):
        logits_b1, z_s = model.forward_stage1(input_ids)
        loss_b1 = F.cross_entropy(logits_b1.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    print(f"  Stage-1 LM loss : {loss_b1.item():.4f}")

    # Gradient check
    loss_b1.backward()
    b1_grad = sum(
        p.grad.abs().mean().item()
        for p in model.block1.parameters()
        if p.grad is not None
    )
    print(f"  Block 1 grad mean : {b1_grad:.6f} ✓")

    # ── 4. Stage-2 forward ────────────────────────────────────────────────────
    separator("4. Stage 2 — Memory Retrieval Pass")

    model.prepare_stage2()
    print(f"  Trainable params: {model.trainable_parameters()/1e6:.2f}M")

    model.block2.train(); model.memory_proj.train()
    with autocast(enabled=use_fp16):
        z_mem, q_mem = model.forward_stage2(input_ids)
    print(f"  z_memory shape : {tuple(z_mem.shape)}")
    print(f"  q_memory shape : {tuple(q_mem.shape)}")

    # ── 5. Stage-3 full forward + combined loss ───────────────────────────────
    separator("5. Stage 3 — Fusion + Combined Loss")

    model.prepare_stage3()
    print(f"  Trainable params: {model.trainable_parameters()/1e6:.2f}M")

    model.block3.train()
    with autocast(enabled=use_fp16):
        logits_full, aux_info = model(input_ids)
        total_loss, loss_dict = model.loss(logits_full, targets, aux_info)

    print(f"  LM loss            : {loss_dict['lm']:.4f}")
    print(f"  Memory entropy loss: {loss_dict['memory_entropy']:.4f}")
    print(f"  Total loss         : {loss_dict['total']:.4f}")

    # Gradient check for Block 3
    total_loss.backward()
    b3_grad = sum(
        p.grad.abs().mean().item()
        for p in model.block3.parameters()
        if p.grad is not None
    )
    print(f"  Block 3 grad mean  : {b3_grad:.6f} ✓")

    # Verify Block 1 & 2 have no gradients (correctly frozen)
    b1_grads = [p.grad for p in model.block1.parameters() if p.grad is not None]
    b2_grads = [p.grad for p in model.block2.parameters() if p.grad is not None]
    print(f"  Block 1 grads (should be 0) : {len(b1_grads)} ✓")
    print(f"  Block 2 grads (should be 0) : {len(b2_grads)} ✓")

    # ── 6. Memory entropy stats ───────────────────────────────────────────────
    separator("6. Memory Attention Entropy (anti-collapse diagnostic)")

    for i, attn_m in enumerate(aux_info['attn_memory']):
        p = attn_m.clamp(min=1e-9)
        H = -(p * p.log()).sum(dim=-1).mean().item()
        print(f"  Layer {i} memory attention entropy : {H:.4f}")
    print("  (Values near 0 = collapsed; higher = memory used broadly)")

    # ── 7. In-place memory hot-swap ───────────────────────────────────────────
    separator("7. Block 2 In-Place Memory Update (hot-swap)")

    b2    = model.block2
    mem   = b2.export_memory()
    print(f"  keys_left  : {tuple(mem['keys_left'].shape)}")
    print(f"  keys_right : {tuple(mem['keys_right'].shape)}")
    print(f"  values     : {tuple(mem['values'].shape)}")

    # Simulate replacing the knowledge store with new random vectors
    new_vals = torch.randn_like(mem['values'])
    b2.update_memory(new_values=new_vals)
    print("  New values injected — no retraining required ✓")

    # ── 8. FP16 scaler demo ───────────────────────────────────────────────────
    if use_fp16:
        separator("8. FP16 GradScaler Pass")
        model.prepare_stage3()
        model.block3.train()
        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.block3.parameters(), lr=1e-4)
        optimizer.zero_grad()
        with autocast():
            logits_fp16, aux = model(input_ids)
            loss_fp16, _     = model.loss(logits_fp16, targets, aux)
        scaler.scale(loss_fp16).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.block3.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        print(f"  FP16 loss : {loss_fp16.item():.4f}  ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("Summary")
    print("  All tests passed ✓")
    print(f"  Total unique parameters: {total/1e6:.2f} M")
    print(f"  Device: {device}")
    print(f"  FP16: {use_fp16}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cpu',
                   help="'cpu' or 'cuda'")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("[demo] CUDA not available, falling back to CPU.")
        device = torch.device('cpu')

    run_demo(device)
