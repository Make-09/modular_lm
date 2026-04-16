"""
stage2_memory.py — Train Block 2 (Knowledge Memory) + MemoryQueryProjection.
=============================================================================

Block 1 is loaded from Stage-1 checkpoint and FROZEN.

Objective
---------
The memory block is trained to produce representations that are useful for
next-token prediction when combined with syntax embeddings.

Specifically we train:
  1. MemoryQueryProjection  (Block-1 output → memory query)
  2. Block2ProductKeyMemory (query → retrieved knowledge vector)
  3. A thin linear "probe head" (temporary, discarded after Stage 2)
     that predicts the next token from z_memory alone.

Why train memory with a probe?
  We want memory to learn to store and retrieve content-rich vectors,
  NOT rely on Block 1 to do the heavy lifting. The probe forces the memory
  output to be independently predictive.

After training : memory_proj + Block 2 weights are saved and frozen.

Usage
-----
  python stage2_memory.py \
      --block1_ckpt checkpoints/block1.pt \
      --save_path   checkpoints/block2.pt \
      --dummy
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.block1_syntax   import Block1SyntaxEngine
from model.block2_memory   import Block2ProductKeyMemory
from model.modular_lm      import MemoryQueryProjection
from utils.data import (
    load_text_dataset, make_dummy_dataset,
    get_dataloader, save_checkpoint, load_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────────
# Temporary probe head
# ──────────────────────────────────────────────────────────────────────────────

class MemoryProbeHead(nn.Module):
    """
    Lightweight probe: z_memory → logits.
    Trained only in Stage 2 to force content-rich memory representations.
    Discarded after Stage 2.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.linear  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, z_memory: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(z_memory))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_stage2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — Knowledge Memory Training")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load Stage-1 config ───────────────────────────────────────────────────
    if not os.path.exists(args.block1_ckpt):
        raise FileNotFoundError(
            f"Block 1 checkpoint not found: {args.block1_ckpt}\n"
            f"Run stage1_train.py first."
        )
    b1_ckpt    = load_checkpoint(args.block1_ckpt, device)
    b1_cfg     = b1_ckpt['config']
    vocab_size = b1_cfg['vocab_size']
    d_model    = b1_cfg['d_model']
    seq_len    = b1_cfg['max_seq_len']

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dummy:
        print("[Stage 2] Using random dummy data.")
        train_ds = make_dummy_dataset(vocab_size, n_tokens=200_000, seq_len=seq_len)
        val_ds   = make_dummy_dataset(vocab_size, n_tokens=10_000,  seq_len=seq_len)
    else:
        train_ds, val_ds, _ = load_text_dataset(args.data_path, seq_len=seq_len)

    train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=1, shuffle=False)

    # ── Block 1 (frozen) ──────────────────────────────────────────────────────
    block1 = Block1SyntaxEngine(
        vocab_size   = vocab_size,
        d_model      = d_model,
        n_heads      = b1_cfg['n_heads'],
        n_layers     = b1_cfg['n_layers'],
        max_seq_len  = seq_len,
        ffn_mult     = 2,
        use_gradient_checkpointing = False,   # frozen, no grad needed
    ).to(device)
    block1.load_state_dict(b1_ckpt['model_state'])
    block1.freeze()
    block1.eval()
    print(f"  Block 1 loaded & frozen ({sum(p.numel() for p in block1.parameters())/1e6:.2f}M)\n")

    # ── Memory modules (trainable) ────────────────────────────────────────────
    mem_proj = MemoryQueryProjection(d_model).to(device)
    block2   = Block2ProductKeyMemory(
        d_model   = d_model,
        d_memory  = args.d_memory,
        num_keys  = args.num_keys,
        top_k     = args.top_k,
        num_heads = args.b2_num_heads,
    ).to(device)
    probe = MemoryProbeHead(d_model, vocab_size).to(device)

    print(f"  Memory projection : {sum(p.numel() for p in mem_proj.parameters())/1e6:.3f}M")
    print(f"  Block 2 (memory)  : {sum(p.numel() for p in block2.parameters())/1e6:.3f}M")
    print(f"  Probe head        : {sum(p.numel() for p in probe.parameters())/1e6:.3f}M\n")

    trainable = list(mem_proj.parameters()) + list(block2.parameters()) + list(probe.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float('inf')

    for epoch in range(args.epochs):
        mem_proj.train(); block2.train(); probe.train()
        epoch_loss = 0.0
        n_steps    = 0
        t0         = time.time()
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                # Block 1 — no gradient
                with torch.no_grad():
                    z_struct = block1(x)                           # (1, T, d_model)

                # Memory path — with gradient
                q_memory = mem_proj(z_struct)                      # (1, T, d_model)
                z_memory = block2(q_memory)                        # (1, T, d_model)
                logits   = probe(z_memory)                         # (1, T, vocab_size)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                ) / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                n_steps += 1

            if args.max_steps_per_epoch and n_steps >= args.max_steps_per_epoch:
                break

        # ── Validation ────────────────────────────────────────────────────────
        mem_proj.eval(); block2.eval(); probe.eval()
        val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    z_struct = block1(x)
                    q_memory = mem_proj(z_struct)
                    z_memory = block2(q_memory)
                    logits   = probe(z_memory)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item(); val_n += 1
                if val_n >= args.max_val_steps:
                    break

        avg_val = val_loss / max(val_n, 1)
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={epoch_loss/max(n_steps*args.grad_accum,1):.4f} | "
            f"val_loss={avg_val:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={time.time()-t0:.1f}s"
        )

        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint({
                'mem_proj_state':  mem_proj.state_dict(),
                'block2_state':    block2.state_dict(),
                'val_loss':        best_val,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model':    d_model,
                    'd_memory':   args.d_memory,
                    'num_keys':   args.num_keys,
                    'top_k':      args.top_k,
                    'num_heads':  args.b2_num_heads,
                    'seq_len':    seq_len,
                },
            }, args.save_path)

    print(f"\n[Stage 2] Training complete. Best val loss: {best_val:.4f}")
    print(f"[Stage 2] Checkpoint → {args.save_path}")
    print("[Stage 2] Block 2 + projection ready to be frozen.\n")
    return mem_proj, block2


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Stage 2: Train Knowledge Memory")
    p.add_argument('--block1_ckpt', type=str, default='checkpoints/block1.pt')
    p.add_argument('--data_path',   type=str, default='data/corpus.txt')
    p.add_argument('--dummy',       action='store_true')
    p.add_argument('--save_path',   type=str, default='checkpoints/block2.pt')
    # Memory config
    p.add_argument('--d_memory',    type=int, default=256)
    p.add_argument('--num_keys',    type=int, default=128)
    p.add_argument('--top_k',       type=int, default=16)
    p.add_argument('--b2_num_heads',type=int, default=4)
    # Training
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--grad_accum',  type=int,   default=8)
    p.add_argument('--max_steps_per_epoch', type=int, default=500)
    p.add_argument('--max_val_steps',       type=int, default=100)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_stage2(args)
