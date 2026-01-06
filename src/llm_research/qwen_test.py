# ============================================================
# Qwen-2.5-Coder-32B EasyDeL TPU Script
# ============================================================

import os
os.environ["PJRT_DEVICE"] = "TPU"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from transformers import AutoTokenizer

from easydel import (
    AutoEasyDelConfig,
    AutoEasyDelModelForCausalLM,
    EasyDelTrainingArguments,
    EasyDelTrainer
)

# ============================================================
# 1. CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
DTYPE = jnp.bfloat16

TP_SIZE = 8      # tensor parallel
DP_SIZE = 4      # data parallel
TOTAL_DEVICES = TP_SIZE * DP_SIZE

MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 64

# ============================================================
# 2. TPU MESH
# ============================================================

devices = jax.devices()
assert len(devices) == TOTAL_DEVICES, (
    f"Expected {TOTAL_DEVICES} TPU devices, got {len(devices)}"
)

mesh = Mesh(
    mesh_utils.create_device_mesh((DP_SIZE, TP_SIZE)),
    axis_names=("data", "model")
)

print(f"✅ TPU Mesh initialized: {mesh}")

# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. LOAD QWEN 2.5 MODEL (JAX / TPU)
# ============================================================

config = AutoEasyDelConfig.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    trust_remote_code=True,
    max_position_embeddings=MAX_SEQ_LEN
)

with mesh:
    model = AutoEasyDelModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        shard_parameters=True,      # <<< CRITICAL
        trust_remote_code=True
    )

print("✅ Qwen-2.5-Coder-32B loaded & sharded")

# ============================================================
# 5. LOGIT-ACCESS FORWARD (RESEARCH SAFE)
# ============================================================

@jax.jit
def forward_with_logits(params, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        params=params,
        deterministic=True,
        output_hidden_states=True
    )
    return outputs.logits, outputs.hidden_states

# ============================================================
# 6. INFERENCE DEMO
# ============================================================

prompt = "Write a high-performance C++ thread pool."

inputs = tokenizer(
    prompt,
    return_tensors="np",
    padding="max_length",
    max_length=512,
    truncation=True
)

input_ids = jnp.array(inputs["input_ids"])
attention_mask = jnp.array(inputs["attention_mask"])

logits, hidden_states = forward_with_logits(
    model.params,
    input_ids,
    attention_mask
)

next_token = jnp.argmax(logits[:, -1], axis=-1)
decoded = tokenizer.decode(next_token.tolist())

print("\n--- Inference Test ---")
print(decoded)
print("----------------------")

# ============================================================
# 7. FINE-TUNING SETUP (LoRA or Full FT)
# ============================================================

training_args = EasyDelTrainingArguments(
    output_dir="./qwen25_32b_ckpts",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    max_steps=1000,
    mesh=mesh,
    optimizer="adamw",
    gradient_checkpointing=True,
    max_sequence_length=MAX_SEQ_LEN,
)

# Example dataset placeholder
# Must return dict with input_ids, attention_mask, labels
train_dataset = None  # <-- plug your dataset here

trainer = EasyDelTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
)

# trainer.train()   # <-- uncomment to train

# ============================================================
# 8. CUSTOM LOGIT MODIFICATION EXAMPLE
# ============================================================

def apply_logit_penalty(logits):
    penalty = jnp.where(logits < -10.0, -5.0, 0.0)
    return logits + penalty

@jax.jit
def forward_with_custom_logits(params, input_ids, attention_mask):
    logits, hidden_states = forward_with_logits(
        params, input_ids, attention_mask
    )
    logits = apply_logit_penalty(logits)
    return logits, hidden_states

print("✅ Ready for fine-tuning & research")
