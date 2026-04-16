"""
finetune_chat.py — Stage 4: Instruction Fine-tuning (Chat)
============================================================
Loads the three pre-trained block checkpoints and fine-tunes the model
on multilingual instruction data (KAZ / RUS / ENG).

Training strategy
-----------------
  • Blocks 1 & 2  : frozen  (preserve syntax + factual knowledge)
  • memory_proj   : optionally unfrozen (--unfreeze_proj)
  • Block 3       : trainable  ← learns to follow instructions

Loss
----
  Standard cross-entropy on ASSISTANT tokens only.
  Memory entropy aux loss is kept active (λ=0.1).

Hardware target: RTX 3050 4 GB (FP16, grad_accum=8, batch=1, seq=256)

Usage
-----
    python finetune_chat.py \\
        --block1_ckpt checkpoints/block1.pt \\
        --block2_ckpt checkpoints/block2.pt \\
        --block3_ckpt checkpoints/block3.pt \\
        --tokenizer   tokenizer/multilingual.model \\
        --train_data  data/processed/train.jsonl \\
        --val_data    data/processed/val.jsonl

    # Quick smoke-test (no data needed):
    python finetune_chat.py --dummy
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import MODEL_CFG, TRAIN_CFG, CKPT
from model.modular_lm import ModularLM
from tokenizer import MultilingualTokenizer
from utils.chat_template import ChatDataset, collate_fn, IGNORE_ID
from utils.data import get_dataloader, save_checkpoint


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block1_ckpt",    default=CKPT["block1"])
    p.add_argument("--block2_ckpt",    default=CKPT["block2"])
    p.add_argument("--block3_ckpt",    default=CKPT["block3"])
    p.add_argument("--tokenizer",      default="tokenizer/multilingual.model")
    p.add_argument("--train_data",     default="data/processed/train.jsonl")
    p.add_argument("--val_data",       default="data/processed/val.jsonl")
    p.add_argument("--output_ckpt",    default=CKPT["chat"])
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",         type=int,   default=TRAIN_CFG["max_epochs"])
    p.add_argument("--lr",             type=float, default=1e-4)   # lower for fine-tune
    p.add_argument("--grad_accum",     type=int,   default=TRAIN_CFG["grad_accum"])
    p.add_argument("--seq_len",        type=int,   default=TRAIN_CFG["seq_len"])
    p.add_argument("--log_every",      type=int,   default=TRAIN_CFG["log_every"])
    p.add_argument("--eval_every",     type=int,   default=TRAIN_CFG["eval_every"])
    p.add_argument("--save_every",     type=int,   default=TRAIN_CFG["save_every"])
    p.add_argument("--unfreeze_proj",  action="store_true",
                   help="Also train memory_proj (slight quality boost)")
    p.add_argument("--dummy",          action="store_true",
                   help="Smoke-test with random data, no files needed")
    return p.parse_args()


# ── Loss (assistant tokens only) ──────────────────────────────────────────────

def chat_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy only on positions where label != IGNORE_ID.
    logits : (B, T, V)
    labels : (B, T)
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=IGNORE_ID,
    )


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(enabled=(device == "cuda")):
            logits, _ = model(x, return_aux=False)
            loss = chat_loss(logits, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    if args.dummy:
        tokenizer = None
        vocab_size = MODEL_CFG["vocab_size"]
    else:
        tokenizer  = MultilingualTokenizer(args.tokenizer)
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer vocab size: {vocab_size}")

    # ── Model ──────────────────────────────────────────────────────────────────
    cfg = {**MODEL_CFG, "vocab_size": vocab_size}
    model = ModularLM(**cfg).to(device)

    # Load checkpoints
    if not args.dummy:
        for attr, path in [
            ("block1", args.block1_ckpt),
            ("block2", args.block2_ckpt),
            ("block3", args.block3_ckpt),
        ]:
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                key  = attr if attr in ckpt else "model_state"
                getattr(model, attr).load_state_dict(ckpt.get(key, ckpt))
                print(f"  Loaded {attr} ← {path}")
            else:
                print(f"  [WARN] {path} not found — starting from scratch")

    # ── Freeze / unfreeze ──────────────────────────────────────────────────────
    model.prepare_stage3()   # Block 1 & 2 frozen, Block 3 trainable
    if args.unfreeze_proj:
        model.memory_proj.unfreeze()
        print("  memory_proj unfrozen")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train / 1e6:.2f} M")

    # ── Data ───────────────────────────────────────────────────────────────────
    if args.dummy:
        # Random dummy batch
        def dummy_loader():
            while True:
                x = torch.randint(0, vocab_size, (1, args.seq_len))
                y = x.clone()
                y[:, :args.seq_len // 2] = IGNORE_ID
                yield x, y

        train_loader = dummy_loader()
        val_loader   = None
        steps_per_epoch = 200
    else:
        train_ds = ChatDataset(args.train_data, tokenizer, max_len=args.seq_len)
        val_ds   = ChatDataset(args.val_data,   tokenizer, max_len=args.seq_len)
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        steps_per_epoch = len(train_loader)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.1,
    )
    total_steps = args.epochs * steps_per_epoch // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr / 10
    )
    scaler = GradScaler(enabled=(args.device == "cuda"))

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    optimizer.zero_grad()

    print("\n" + "=" * 60)
    print("Stage 4 — Instruction Fine-tuning (Chat)")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if args.dummy and step >= steps_per_epoch:
                break

            x, y = batch if not args.dummy else next(train_loader)
            if not args.dummy:
                x, y = x.to(device), y.to(device)
            else:
                x, y = x.to(device), y.to(device)

            with autocast(enabled=(args.device == "cuda")):
                logits, aux_info = model(x, return_aux=True)
                lm_loss  = chat_loss(logits, y)
                aux_loss = model.memory_entropy_loss(aux_info) if aux_info else 0.0
                loss     = lm_loss + model.memory_aux_lambda * aux_loss
                loss     = loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"  epoch {epoch} | step {global_step:5d} | "
                        f"loss {epoch_loss / (step + 1):.4f} | lr {lr_now:.2e}"
                    )

                if val_loader and global_step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  [VAL] step {global_step} | val_loss {val_loss:.4f}")

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        {"model_state": model.block3.state_dict(),
                         "global_step": global_step,
                         "epoch": epoch},
                        args.output_ckpt.replace(".pt", f"_step{global_step}.pt"),
                    )

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.0f}s | "
              f"avg loss {epoch_loss / max(steps_per_epoch, 1):.4f}")

    # ── Save final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(
        {
            "block3":       model.block3.state_dict(),
            "memory_proj":  model.memory_proj.state_dict(),
            "vocab_size":   vocab_size,
            "model_cfg":    cfg,
            "epoch":        args.epochs,
        },
        args.output_ckpt,
    )
    print(f"\n✓ Chat model saved → {args.output_ckpt}")

    if args.dummy:
        print("Smoke-test complete! Re-run without --dummy on real data.")


if __name__ == "__main__":
    main()
