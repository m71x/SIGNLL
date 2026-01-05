import os
os.environ["PJRT_DEVICE"] = "TPU"

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# ----------------------------------------------------------------------
# LOAD TOKENIZER + MODEL
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = FlaxAutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=jnp.bfloat16,
    trust_remote_code=True,
)

# ----------------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."

inputs = tokenizer(prompt, return_tensors="np")
input_ids = jnp.array(inputs["input_ids"])

# ----------------------------------------------------------------------
# PREFILL
# ----------------------------------------------------------------------
def prefill(input_ids):
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        deterministic=True,
    )
    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)
    return outputs.past_key_values, next_token

past_kv, next_token = prefill(input_ids)

# ----------------------------------------------------------------------
# DECODE STEP (STATIC)
# ----------------------------------------------------------------------
def decode_step(carry, _):
    past_kv, token = carry

    outputs = model(
        input_ids=token[:, None],
        past_key_values=past_kv,
        use_cache=True,
        deterministic=True,
    )

    next_token = jnp.argmax(outputs.logits[:, -1], axis=-1)

    return (outputs.past_key_values, next_token), next_token

# ----------------------------------------------------------------------
# COMPILED DECODE LOOP
# ----------------------------------------------------------------------
(past_kv, last_token), generated = jax.lax.scan(
    decode_step,
    (past_kv, next_token),
    xs=None,
    length=MAX_NEW_TOKENS - 1,
)

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------
all_tokens = jnp.concatenate(
    [input_ids, next_token[:, None], generated.T],
    axis=1,
)

print(tokenizer.decode(np.array(all_tokens[0]), skip_special_tokens=True))
