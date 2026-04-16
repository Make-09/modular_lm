"""
config.py — Central configuration for ModularLM.

All training scripts import from here so you only change things in one place.
"""

# ── Tokenizer ──────────────────────────────────────────────────────────────────
TOKENIZER_MODEL = "tokenizer/multilingual.model"   # SentencePiece model path
VOCAB_SIZE       = 32_000                           # Must match model vocab_size

# ── Model (defaults, ~49 M params) ────────────────────────────────────────────
MODEL_CFG = dict(
    vocab_size  = VOCAB_SIZE,
    d_model     = 512,
    max_seq_len = 256,
    dropout     = 0.1,
    # Block 1
    b1_n_heads  = 8,
    b1_n_layers = 6,
    b1_ffn_mult = 2,
    # Block 2
    b2_d_memory  = 256,
    b2_num_keys  = 128,
    b2_top_k     = 16,
    b2_num_heads = 4,
    # Block 3
    b3_n_heads        = 8,
    b3_n_layers       = 4,
    b3_ffn_mult       = 2,
    b3_struct_dropout = 0.2,
    # Aux loss
    memory_aux_lambda      = 0.1,
    memory_target_entropy  = 2.0,
    # Memory
    use_gradient_checkpointing = True,
)

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_CFG = dict(
    batch_size  = 1,        # RTX 3050 4 GB constraint
    grad_accum  = 8,        # effective batch = 8
    seq_len     = 256,
    lr          = 3e-4,
    weight_decay = 0.1,
    max_epochs  = 3,
    warmup_steps = 200,
    clip_grad   = 1.0,
    fp16        = True,
    log_every   = 50,       # steps
    eval_every  = 500,
    save_every  = 1000,
)

# ── Checkpoints ────────────────────────────────────────────────────────────────
CKPT = dict(
    block1 = "checkpoints/block1.pt",
    block2 = "checkpoints/block2.pt",
    block3 = "checkpoints/block3.pt",
    chat   = "checkpoints/chat.pt",
)

# ── Data ───────────────────────────────────────────────────────────────────────
DATA_DIR = "data/processed"

# ── Chat template tokens ───────────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are a helpful assistant. "
    "You can speak Kazakh (қазақша), Russian (русский), and English. "
    "Respond in the same language the user writes in.\n"
    "Сіз қазақша, орысша және ағылшынша сөйлей аласыз.\n"
    "Вы говорите на казахском, русском и английском языках."
)
