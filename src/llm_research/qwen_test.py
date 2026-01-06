# ============================================================================
# MUST COME FIRST — ENVIRONMENT SETUP
# ============================================================================
import os

CACHE_DIR = "/dev/shm/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

os.environ["TMPDIR"] = CACHE_DIR
os.environ["TEMP"] = CACHE_DIR
os.environ["TMP"] = CACHE_DIR

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PJRT_DEVICE"] = "TPU"

# ============================================================================
# IMPORTS
# ============================================================================
import jax

try:
    jax.distributed.initialize()
except Exception:
    print("JAX distributed already initialized or not needed.")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis
from transformers import AutoTokenizer, GenerationConfig, AutoConfig

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 1900
CONTEXT_LENGTH = 4096  # Increased to handle prompt + output safely

DP_SIZE = 1
FSDP_SIZE = 1
TP_SIZE = 8
SP_SIZE = 4

# ============================================================================
# 1. MESH & SHARDING (MOVED TO TOP)
# ============================================================================
# CRITICAL FIX: The mesh must be defined BEFORE loading the model so JAX
# knows how to distribute the weights across the TPUs immediately.
device_mesh = mesh_utils.create_device_mesh(
    (DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE)
)

mesh = Mesh(device_mesh, axis_names=("dp", "fsdp", "tp", "sp"))

# ============================================================================
# TOKENIZER & CONFIG
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

config.max_position_embeddings = CONTEXT_LENGTH

# ============================================================================
# 2. LOAD MODEL (WRAPPED IN MESH)
# ============================================================================
print("Loading model...")
# CRITICAL FIX: Wrap loading in 'with mesh:' to enable immediate sharding
with mesh:
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        config_kwargs={
            "max_position_embeddings": CONTEXT_LENGTH,
            "max_sequence_length": CONTEXT_LENGTH,
            "use_scan_mlp": False,
        },
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        # Map model dimensions to the mesh axes defined above
        sharding_axis_dims=(DP_SIZE, FSDP_SIZE, TP_SIZE, SP_SIZE),
        sharding_axis_names=("dp", "fsdp", "tp", "sp"),
        partition_axis=PartitionAxis(),
        shard_attention_computation=True,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
print("Model loaded and sharded successfully.")

# ============================================================================
# INPUT PREPARATION
# ============================================================================
prompt = "Write a python program that shows an example of the diamond problem in inheritance."

formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="jax",
    padding=False,
    truncation=True,
)

input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", jnp.ones_like(input_ids))

# Calculate padding so input length is divisible by SP_SIZE
# (This ensures the input tensor can be evenly sharded across devices)
current_len = input_ids.shape[1]
pad = (SP_SIZE - current_len % SP_SIZE) % SP_SIZE

if pad > 0:
    pad_ids = jnp.full(
        (input_ids.shape[0], pad),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
    )
    pad_mask = jnp.zeros(
        (attention_mask.shape[0], pad),
        dtype=attention_mask.dtype,
    )
    input_ids = jnp.concatenate([pad_ids, input_ids], axis=1)
    attention_mask = jnp.concatenate([pad_mask, attention_mask], axis=1)

# ============================================================================
# SHARD INPUTS
# ============================================================================
# Define how inputs should be split:
# Batch dimension -> None (Replicated)
# Sequence dimension -> "sp" (Sharded across 4 devices)
input_sharding = NamedSharding(mesh, P(None, "sp"))

input_ids = jax.device_put(input_ids, input_sharding)
attention_mask = jax.device_put(attention_mask, input_sharding)

# ============================================================================
# GENERATION
# ============================================================================
generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

if not hasattr(generation_config, "forced_decoder_ids"):
    generation_config.forced_decoder_ids = None


print("Starting generation...")
with mesh:
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

output_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n=== GENERATED TEXT ===\n")
print(generated_text)