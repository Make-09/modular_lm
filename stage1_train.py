"""
stage1_train.py — Train Block 1 (Syntax Engine) on next-token prediction.
==========================================================================

Objective : Causal language modelling (cross-entropy on next token).
After training : Block 1 weights are frozen. Checkpoint saved to disk.

Hardware strategy
-----------------
  * FP16 mixed precision via torch.cuda.amp
  * Gradient accumulation (default --grad_accum 8)
  * Gradient checkpointing inside Block1SyntaxEngine
  * batch_size = 1 (4 GB VRAM constraint)

Usage
-----
  # With a text file:
  python stage1_train.py --data_path data/corpus.txt

  # Smoke-test with dummy data:
  python stage1_train.py --dummy

  # Resume from checkpoint:
  python stage1_train.py --data_path data/corpus.txt --resume checkpoints/block1.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# ── Make sure the project root is importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.block1_syntax import Block1SyntaxEngine
from utils.data import (
    load_text_dataset, make_dummy_dataset,
    get_dataloader, save_checkpoint, load_checkpoint,
)

# Optional: use MultilingualTokenizer if available
try:
    from tokenizer import MultilingualTokenizer as _MLTok
    _HAS_ML_TOK = True
except ImportError:
    _HAS_ML_TOK = False


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_stage1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Syntax Engine Training")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dummy:
        print("[Stage 1] Using random dummy data.")
        vocab_size = args.vocab_size
        train_ds   = make_dummy_dataset(vocab_size, n_tokens=200_000, seq_len=args.seq_len)
        val_ds     = make_dummy_dataset(vocab_size, n_tokens=10_000,  seq_len=args.seq_len)
        tok        = None
    else:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        # Prefer multilingual BPE tokenizer if a model file is present
        ml_tok_path = getattr(args, 'tokenizer', 'tokenizer/multilingual.model')
        if _HAS_ML_TOK and ml_tok_path and os.path.exists(ml_tok_path):
            import torch as _torch
            ml_tok = _MLTok(ml_tok_path)
            with open(args.data_path, 'r', encoding='utf-8') as _f:
                _text = _f.read()
            _ids = _torch.tensor(ml_tok.encode(_text), dtype=_torch.long)
            _n   = int(len(_ids) * 0.9)
            from utils.data import TokenDataset
            train_ds   = TokenDataset(_ids[:_n],  args.seq_len)
            val_ds     = TokenDataset(_ids[_n:],  args.seq_len)
            vocab_size = ml_tok.vocab_size
            tok = ml_tok
            print(f"  [MultilingualTokenizer] vocab={vocab_size}")
        else:
            train_ds, val_ds, tok = load_text_dataset(
                args.data_path, seq_len=args.seq_len
            )
            vocab_size = tok.vocab_size

    print(f"  Vocab size   : {vocab_size:,}")
    print(f"  Train tokens : {len(train_ds):,} windows")
    print(f"  Val tokens   : {len(val_ds):,} windows\n")

    train_loader = get_dataloader(train_ds, batch_size=1, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=1, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Block1SyntaxEngine(
        vocab_size   = vocab_size,
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        max_seq_len  = args.seq_len,
        ffn_mult     = 2,
        dropout      = 0.1,
        use_gradient_checkpointing = True,
    ).to(device)

    # Optional: resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[Stage 1] Resumed from epoch {start_epoch}")

    n_params = model.count_parameters()
    print(f"  Block 1 parameters : {n_params/1e6:.2f} M\n")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0
        t0         = time.time()
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                logits, _ = model.forward_lm(x)                   # (1, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                ) / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                n_steps += 1

            # Limit steps per epoch if requested (fast iteration)
            if args.max_steps_per_epoch and n_steps >= args.max_steps_per_epoch:
                break

        avg_train_loss = epoch_loss / max(n_steps * args.grad_accum, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_n    = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    logits, _ = model.forward_lm(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        y.reshape(-1),
                    )
                val_loss += loss.item()
                val_n    += 1
                if val_n >= args.max_val_steps:
                    break

        avg_val_loss = val_loss / max(val_n, 1)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={elapsed:.1f}s"
        )

        # ── Save best ──────────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    best_val_loss,
                'config': {
                    'vocab_size':  vocab_size,
                    'd_model':     args.d_model,
                    'n_heads':     args.n_heads,
                    'n_layers':    args.n_layers,
                    'max_seq_len': args.seq_len,
                },
            }, args.save_path)

    print(f"\n[Stage 1] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"[Stage 1] Checkpoint → {args.save_path}")
    print("[Stage 1] Block 1 is now ready to be frozen.\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Stage 1: Train Block 1 Syntax Engine")
    p.add_argument('--data_path',  type=str,   default='data/corpus.txt')
    p.add_argument('--dummy',      action='store_true',  help='Use random dummy data')
    p.add_argument('--save_path',  type=str,   default='checkpoints/block1.pt')
    p.add_argument('--resume',     type=str,   default='')
    # Architecture
    p.add_argument('--vocab_size', type=int,   default=256,  help='Used only with --dummy')
    p.add_argument('--d_model',    type=int,   default=512)
    p.add_argument('--n_heads',    type=int,   default=8)
    p.add_argument('--n_layers',   type=int,   default=6)
    p.add_argument('--seq_len',    type=int,   default=256)
    # Training
    p.add_argument('--epochs',           type=int,   default=10)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--grad_accum',       type=int,   default=8)
    p.add_argument('--max_steps_per_epoch', type=int, default=500)
    p.add_argument('--max_val_steps',    type=int,   default=100)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_stage1(args)
