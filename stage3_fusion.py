"""
stage3_fusion.py — Train Block 3 (Fusion / Working Memory).
============================================================

Blocks 1 & 2 are loaded from their respective checkpoints and FROZEN.
Only Block 3 (FusionModule) is trained.

Objective
---------
  L_total = L_lm  +  λ · L_memory_entropy

  L_lm              : standard next-token prediction (cross-entropy)
  L_memory_entropy  : anti-collapse auxiliary loss
                      relu(H_target − H_actual) averaged over fusion layers
                      → penalises low entropy in memory cross-attention
                      → ensures Block 3 cannot ignore z_memory

Usage
-----
  python stage3_fusion.py \
      --block1_ckpt checkpoints/block1.pt \
      --block2_ckpt checkpoints/block2.pt \
      --save_path   checkpoints/block3.pt \
      --dummy
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.block1_syntax  import Block1SyntaxEngine
from model.block2_memory  import Block2ProductKeyMemory
from model.block3_fusion  import Block3FusionModule
from model.modular_lm     import MemoryQueryProjection, ModularLM
from utils.data import (
    load_text_dataset, make_dummy_dataset,
    get_dataloader, save_checkpoint, load_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_stage3(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STAGE 3 — Fusion Module Training")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load configs from checkpoints ─────────────────────────────────────────
    b1_ckpt  = load_checkpoint(args.block1_ckpt, device)
    b2_ckpt  = load_checkpoint(args.block2_ckpt, device)
    b1_cfg   = b1_ckpt['config']
    b2_cfg   = b2_ckpt['config']

    vocab_size = b1_cfg['vocab_size']
    d_model    = b1_cfg['d_model']
    seq_len    = b1_cfg['max_seq_len']

    assert b2_cfg['vocab_size'] == vocab_size, "Vocab mismatch between Block 1 and 2"
    assert b2_cfg['d_model']    == d_model,    "d_model mismatch between Block 1 and 2"

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dummy:
        print("[Stage 3] Using random dummy data.")
        train_ds = make_dummy_dataset(vocab_size, 200_000, seq_len)
        val_ds   = make_dummy_dataset(vocab_size,  10_000, seq_len)
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
        use_gradient_checkpointing = False,
    ).to(device)
    block1.load_state_dict(b1_ckpt['model_state'])
    block1.freeze(); block1.eval()
    print(f"  Block 1 loaded & frozen ({sum(p.numel() for p in block1.parameters())/1e6:.2f}M)")

    # ── Memory projection (frozen) ────────────────────────────────────────────
    mem_proj = MemoryQueryProjection(d_model).to(device)
    mem_proj.load_state_dict(b2_ckpt['mem_proj_state'])
    mem_proj.freeze(); mem_proj.eval()

    # ── Block 2 (frozen) ──────────────────────────────────────────────────────
    block2 = Block2ProductKeyMemory(
        d_model   = d_model,
        d_memory  = b2_cfg['d_memory'],
        num_keys  = b2_cfg['num_keys'],
        top_k     = b2_cfg['top_k'],
        num_heads = b2_cfg['num_heads'],
    ).to(device)
    block2.load_state_dict(b2_ckpt['block2_state'])
    block2.freeze(); block2.eval()
    print(f"  Block 2 loaded & frozen ({sum(p.numel() for p in block2.parameters())/1e6:.2f}M)\n")

    # ── Block 3 (trainable) ───────────────────────────────────────────────────
    block3 = Block3FusionModule(
        vocab_size    = vocab_size,
        d_model       = d_model,
        n_heads       = args.n_heads,
        n_layers      = args.n_layers,
        max_seq_len   = seq_len,
        ffn_mult      = 2,
        dropout       = 0.1,
        struct_dropout = args.struct_dropout,
        use_gradient_checkpointing = True,
        shared_lm_head = block1.lm_head,     # share embedding weights
    ).to(device)
    print(f"  Block 3 (trainable): {block3.count_parameters()/1e6:.2f}M\n")

    # Optional: resume Block 3
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, device)
        block3.load_state_dict(ckpt['block3_state'])
        print(f"  Block 3 resumed from {args.resume}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        block3.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── Auxiliary loss weight ──────────────────────────────────────────────────
    aux_lambda        = args.aux_lambda
    target_entropy    = args.target_entropy

    def memory_entropy_loss(aux_info):
        """Penalise low-entropy memory attention (prevents collapse)."""
        losses = []
        for attn_m in aux_info['attn_memory']:
            p = attn_m.clamp(min=1e-9)
            H = -(p * p.log()).sum(dim=-1).mean()
            losses.append(F.relu(target_entropy - H))
        return sum(losses) / max(len(losses), 1)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float('inf')

    for epoch in range(args.epochs):
        block3.train()
        epoch_lm   = 0.0
        epoch_aux  = 0.0
        n_steps    = 0
        t0         = time.time()
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                # Frozen upstream
                with torch.no_grad():
                    z_struct = block1(x)
                    q_memory = mem_proj(z_struct)
                    z_memory = block2(q_memory)

                # Block 3 — with gradient
                logits, aux_info = block3(z_struct, z_memory)

                # LM loss
                lm_loss  = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                )
                # Auxiliary anti-collapse loss
                aux_loss  = memory_entropy_loss(aux_info)
                total_loss = (lm_loss + aux_lambda * aux_loss) / args.grad_accum

            scaler.scale(total_loss).backward()
            epoch_lm  += lm_loss.item()
            epoch_aux += aux_loss.item()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(block3.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                n_steps += 1

            if args.max_steps_per_epoch and n_steps >= args.max_steps_per_epoch:
                break

        # ── Validation ────────────────────────────────────────────────────────
        block3.eval()
        val_lm = 0.0; val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    z_struct = block1(x)
                    q_memory = mem_proj(z_struct)
                    z_memory = block2(q_memory)
                    logits, _ = block3(z_struct, z_memory)
                    lm_loss   = F.cross_entropy(
                        logits.reshape(-1, vocab_size), y.reshape(-1)
                    )
                val_lm += lm_loss.item(); val_n += 1
                if val_n >= args.max_val_steps:
                    break

        avg_train_lm  = epoch_lm  / max(n_steps, 1)
        avg_train_aux = epoch_aux / max(n_steps, 1)
        avg_val_lm    = val_lm    / max(val_n, 1)

        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"lm={avg_train_lm:.4f} aux={avg_train_aux:.4f} | "
            f"val_lm={avg_val_lm:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"t={time.time()-t0:.1f}s"
        )

        if avg_val_lm < best_val:
            best_val = avg_val_lm
            save_checkpoint({
                'block3_state': block3.state_dict(),
                'val_loss':     best_val,
                'config': {
                    'vocab_size':   vocab_size,
                    'd_model':      d_model,
                    'n_heads':      args.n_heads,
                    'n_layers':     args.n_layers,
                    'seq_len':      seq_len,
                    'struct_dropout': args.struct_dropout,
                },
            }, args.save_path)

    print(f"\n[Stage 3] Training complete. Best val LM loss: {best_val:.4f}")
    print(f"[Stage 3] Checkpoint → {args.save_path}\n")
    return block3


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Stage 3: Train Fusion Module")
    p.add_argument('--block1_ckpt', type=str, default='checkpoints/block1.pt')
    p.add_argument('--block2_ckpt', type=str, default='checkpoints/block2.pt')
    p.add_argument('--data_path',   type=str, default='data/corpus.txt')
    p.add_argument('--dummy',       action='store_true')
    p.add_argument('--save_path',   type=str, default='checkpoints/block3.pt')
    p.add_argument('--resume',      type=str, default='')
    # Block 3 architecture
    p.add_argument('--n_heads',         type=int,   default=8)
    p.add_argument('--n_layers',        type=int,   default=4)
    p.add_argument('--struct_dropout',  type=float, default=0.2)
    # Training
    p.add_argument('--epochs',          type=int,   default=10)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--grad_accum',      type=int,   default=8)
    p.add_argument('--aux_lambda',      type=float, default=0.1)
    p.add_argument('--target_entropy',  type=float, default=2.0)
    p.add_argument('--max_steps_per_epoch', type=int, default=500)
    p.add_argument('--max_val_steps',       type=int, default=100)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_stage3(args)
