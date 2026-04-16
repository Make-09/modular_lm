"""
chat.py — Interactive CLI Chat
================================
Load the trained ModularLM and chat with it in Kazakh, Russian, or English.

Usage
-----
    python chat.py
    python chat.py --device cpu
    python chat.py --block3_ckpt checkpoints/chat.pt --max_new_tokens 200

Commands inside the chat
------------------------
    /quit  or  /exit   — exit
    /reset             — clear conversation history
    /lang kk|ru|en     — hint the model to respond in a specific language
    /info              — show model info
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from config import MODEL_CFG, CKPT
from model.modular_lm import ModularLM
from tokenizer import MultilingualTokenizer
from utils.chat_template import SYSTEM_PROMPTS, DEFAULT_SYSTEM


# ── Sampling ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:         ModularLM,
    input_ids:     torch.Tensor,        # (1, T)
    max_new_tokens: int    = 200,
    temperature:   float   = 0.7,
    top_p:         float   = 0.9,
    top_k:         int     = 50,
    eos_id:        int     = 3,
    end_id:        int     = 7,
    device:        torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Auto-regressive generation with top-p + top-k sampling.
    Stops at <|end|> or [EOS].
    Returns generated token IDs (without the prompt).
    """
    model.eval()
    generated = []
    x = input_ids.to(device)
    max_len = model.block1.max_seq_len   # hard limit from architecture

    for _ in range(max_new_tokens):
        # Truncate context to fit sequence length limit
        x_in = x[:, -max_len:]

        logits, _ = model(x_in, return_aux=False)
        next_logits = logits[0, -1, :]           # (vocab,)

        # Temperature
        next_logits = next_logits / max(temperature, 1e-8)

        # Top-k
        if top_k > 0:
            top_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < top_vals[-1]] = float("-inf")

        # Top-p (nucleus)
        probs = F.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff    = (cum_probs - sorted_probs) > top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()

        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
        tok_id = next_token.item()

        if tok_id in (eos_id, end_id):
            break

        generated.append(tok_id)
        x = torch.cat([x, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return torch.tensor(generated, dtype=torch.long)


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(args, tokenizer: MultilingualTokenizer, device: torch.device):
    cfg = {**MODEL_CFG, "vocab_size": tokenizer.vocab_size}
    model = ModularLM(**cfg).to(device)

    ckpts_to_load = [
        ("block1",      args.block1_ckpt),
        ("block2",      args.block2_ckpt),
        ("block3",      args.block3_ckpt or args.chat_ckpt),
    ]

    for attr, path in ckpts_to_load:
        if path and os.path.exists(path):
            ckpt  = torch.load(path, map_location=device)
            state = ckpt.get(attr) or ckpt.get("model_state") or ckpt
            try:
                getattr(model, attr).load_state_dict(state)
                print(f"  ✓ {attr} ← {path}")
            except Exception as e:
                print(f"  [WARN] Could not load {attr}: {e}")
        else:
            print(f"  [WARN] Checkpoint not found: {path}")

    model.eval()
    return model


# ── Chat loop ──────────────────────────────────────────────────────────────────

def run_chat(model, tokenizer, args, device):
    print("\n" + "=" * 60)
    print("  ModularLM Chat  |  KAZ · RUS · ENG")
    print("  Commands: /quit  /reset  /lang kk|ru|en  /info")
    print("=" * 60 + "\n")

    system = DEFAULT_SYSTEM
    history = []    # list of (role, text) tuples

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit"):
                print("Goodbye!")
                break
            elif cmd[0] == "/reset":
                history.clear()
                print("[History cleared]")
            elif cmd[0] == "/lang" and len(cmd) > 1:
                lang = cmd[1]
                system = SYSTEM_PROMPTS.get(lang, DEFAULT_SYSTEM)
                history.clear()
                print(f"[Language hint set to: {lang}, history cleared]")
            elif cmd[0] == "/info":
                total = sum(p.numel() for p in model.parameters())
                print(f"[Model params: {total / 1e6:.1f} M | "
                      f"Vocab: {tokenizer.vocab_size} | "
                      f"Device: {device}]")
            else:
                print(f"[Unknown command: {user_input}]")
            continue

        # ── Build prompt ──────────────────────────────────────────────────────
        turns = list(history) + [("user", user_input)]
        # Keep context window: drop oldest turns if too long
        while True:
            input_ids = tokenizer.encode_chat(
                turns=turns,
                system=system,
                add_generation_prompt=True,
            )
            if len(input_ids) <= model.block1.max_seq_len - args.max_new_tokens:
                break
            if len(turns) > 1:
                turns = turns[1:]    # drop oldest turn
            else:
                break

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # ── Generate ──────────────────────────────────────────────────────────
        print("Assistant: ", end="", flush=True)
        try:
            gen_ids = generate(
                model, input_tensor,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                top_p          = args.top_p,
                top_k          = args.top_k,
                device         = device,
            )
            response = tokenizer.decode(gen_ids.tolist())
        except Exception as e:
            response = f"[Generation error: {e}]"

        print(response)
        print()

        # Update history
        history.append(("user",      user_input))
        history.append(("assistant", response))


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ModularLM Chat")
    p.add_argument("--block1_ckpt",    default=CKPT["block1"])
    p.add_argument("--block2_ckpt",    default=CKPT["block2"])
    p.add_argument("--block3_ckpt",    default=None,
                   help="Stage 3 checkpoint (before fine-tune)")
    p.add_argument("--chat_ckpt",      default=CKPT["chat"],
                   help="Chat fine-tuned checkpoint (preferred)")
    p.add_argument("--tokenizer",      default="tokenizer/multilingual.model")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--top_k",          type=int,   default=50)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading tokenizer from {args.tokenizer} …")
    tokenizer = MultilingualTokenizer(args.tokenizer)

    print(f"Loading model on {device} …")
    model = load_model(args, tokenizer, device)

    run_chat(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
