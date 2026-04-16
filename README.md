# ModularLM — Multilingual Three-Block Language Model

> **~50 M parameter** experimental LM with explicit separation of syntax, factual knowledge, and reasoning.  
> Supports **Kazakh 🇰🇿 · Russian 🇷🇺 · English 🇬🇧** out of the box.

---

## Architecture at a Glance

```
input_ids  (B, T)
     │
     ▼  ══════════════════════════════════════
     │         BLOCK 1 — Syntax Engine  (~29 M)
     │   token_emb + pos_emb
     │   → 6 × SyntaxLayer (CausalSelfAttn + ReducedFFN)
     ▼  z_struct  (B, T, 512)
     │
     ├── Memory Query Projection (Linear → GELU → Linear → LN)
     ▼  q_memory  (B, T, 512)
     │
     ▼  ══════════════════════════════════════
     │         BLOCK 2 — Product-Key Memory  (~4.5 M)
     │   top-k² retrieval across H × K² slots
     ▼  z_memory  (B, T, 512)
     │
     └─────────────────────────────┐
                                   ▼  ══════════════════
                              BLOCK 3 — Fusion  (~15 M)
                         4 × FusionLayer:
                           self-attn + cross-attn(struct)
                                   + cross-attn(memory)
                                   + BottleneckFFN
                                   ▼
                              logits  (B, T, V)
```

| Block | Role | Params | Frozen after |
|-------|------|--------|--------------|
| 1 — Syntax Engine | Grammar, ordering, token structure | ~29 M | Stage 1 |
| 2 — Product-Key Memory | Facts, entities, associations | ~4.5 M | Stage 2 |
| 3 — Fusion Module | Reasoning, composition, output | ~15 M | — |
| Chat fine-tune | Instruction following (KAZ/RUS/ENG) | Block 3 only | — |

---

## Quick Start

### 1 — Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/modular_lm
cd modular_lm
pip install -r requirements.txt
```

### 2 — Download training data

```bash
python data/prepare_data.py
# Downloads ~120 K instruction examples (KAZ · RUS · ENG) + pretrain text
# Output: data/processed/train.jsonl  val.jsonl  pretrain.txt
```

### 3 — Train the tokenizer

```bash
python tokenizer/train_tokenizer.py
# Trains a 32 000-piece multilingual BPE on the downloaded text
# Output: tokenizer/multilingual.model
```

### 4 — Train the model (three stages)

```bash
# Stage 1 — Syntax Engine
python stage1_train.py \
    --data_path data/processed/pretrain.txt \
    --tokenizer tokenizer/multilingual.model

# Stage 2 — Knowledge Memory
python stage2_memory.py --block1_ckpt checkpoints/block1.pt

# Stage 3 — Fusion Module
python stage3_fusion.py \
    --block1_ckpt checkpoints/block1.pt \
    --block2_ckpt checkpoints/block2.pt
```

### 5 — Instruction fine-tuning (chat)

```bash
python finetune_chat.py \
    --block1_ckpt checkpoints/block1.pt \
    --block2_ckpt checkpoints/block2.pt \
    --block3_ckpt checkpoints/block3.pt \
    --tokenizer   tokenizer/multilingual.model \
    --train_data  data/processed/train.jsonl \
    --val_data    data/processed/val.jsonl
```

### 6 — Chat!

```bash
python chat.py
# /lang kk   — hint Kazakh responses
# /lang ru   — hint Russian responses
# /lang en   — hint English responses
# /reset     — clear history
# /quit      — exit
```

---

## Smoke-test (no data needed)

```bash
python run_demo.py                # architecture test, runs on CPU
python finetune_chat.py --dummy   # training loop test
```

---

## Hardware Requirements

| Setup | Works? | Notes |
|-------|--------|-------|
| RTX 3050 4 GB | ✅ | Default settings (fp16, batch=1, seq=256) |
| RTX 3060+ 8 GB | ✅ | Can increase seq_len to 512 |
| CPU only | ✅ (slow) | Disable fp16 |
| Google Colab (T4) | ✅ | Free tier sufficient |

Estimated training times on RTX 3050 Mobile:

| Stage | Time |
|-------|------|
| Stage 1 (Syntax) | ~1.5 h |
| Stage 2 (Memory) | ~0.5 h |
| Stage 3 (Fusion) | ~1 h   |
| Chat fine-tune   | ~2 h   |

---

## Language Support

A 32 000-piece SentencePiece BPE tokenizer is trained on a balanced mix of Kazakh, Russian, and English text. The system prompt instructs the model to respond in the user's language automatically. Use `/lang kk`, `/lang ru`, or `/lang en` in the chat to override.

---

## Chat Format

```
[BOS] <|system|> You are a helpful assistant... <|end|>
      <|user|>   Сәлем! Сіз кім? <|end|>
      <|assistant|> Мен — ModularLM... <|end|>
[EOS]
```

Loss is computed **only on the assistant tokens** during fine-tuning.

---

## Project Structure

```
modular_lm/
├── model/                     Core architecture (3 blocks)
├── tokenizer/                 Multilingual BPE tokenizer
├── data/                      Dataset download & preparation
├── utils/                     Training utilities + chat template
├── config.py                  Central hyperparameters
├── stage1_train.py            Train Block 1
├── stage2_memory.py           Train Block 2
├── stage3_fusion.py           Train Block 3
├── finetune_chat.py           Instruction fine-tuning (Stage 4)
├── chat.py                    Interactive CLI chat
└── run_demo.py                Architecture smoke-test
```

---

## License

MIT
