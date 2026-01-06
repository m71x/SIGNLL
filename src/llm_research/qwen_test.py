import os
import jax
import jax.numpy as jnp
from easydel import (
    AutoShardAndGatherFunctions,
    EasyDeLFlaxPretrainedModel,
)
from fjformer import GenerateRNG
from transformers import AutoTokenizer, AutoConfig
import flax

# Ensure TPU is used
os.environ["PJRT_DEVICE"] = "TPU"

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_NEW_TOKENS = 30

# TPU MESH CONFIGURATION (32 Chips)
FSDP_SIZE = 4  # Data Parallelism
TP_SIZE = 8    # Tensor Parallelism
SP_SIZE = 1    # Sequence Parallelism

# ----------------------------------------------------------------------
# 2. LOAD TOKENIZER
# ----------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------------
# 3. SETUP MESH AND SHARDING
# ----------------------------------------------------------------------
print(f"Setting up mesh with DP={FSDP_SIZE}, TP={TP_SIZE}, SP={SP_SIZE}")

from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

# Create device mesh
devices = jax.devices()
print(f"Available devices: {len(devices)}")

device_mesh = mesh_utils.create_device_mesh((FSDP_SIZE, TP_SIZE, SP_SIZE))
mesh = Mesh(device_mesh, axis_names=('dp', 'fsdp', 'tp'))

print(f"Mesh created: {mesh}")

# ----------------------------------------------------------------------
# 4. LOAD MODEL WITH EASYDEL
# ----------------------------------------------------------------------
print(f"\nLoading Model...")

# Load config
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

# EasyDeL 0.2.0.2 uses this method
model = EasyDeLFlaxPretrainedModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    precision=jax.lax.Precision.DEFAULT,
    auto_shard_model=True,
    sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
    sharding_axis_names=('dp', 'fsdp', 'tp'),
    mesh=mesh,
    config_kwargs={
        "gradient_checkpointing": "",
        "use_scan_mlp": False,
        "scan_mlp_chunk_size": 1024,
    },
    trust_remote_code=True,
    from_pt=True,
)

print("Model loaded successfully!")

# Get the shard functions for this model
shard_fns = AutoShardAndGatherFunctions.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    sharding_axis_dims=(FSDP_SIZE, TP_SIZE, SP_SIZE),
    sharding_axis_names=('dp', 'fsdp', 'tp'),
    mesh=mesh,
    trust_remote_code=True,
)

# Shard the parameters
params = model.params
if hasattr(shard_fns, 'shard_params'):
    print("Sharding parameters across TPUs...")
    params = shard_fns.shard_params(params)

# ----------------------------------------------------------------------
# 5. PREPARE INPUT
# ----------------------------------------------------------------------
prompt = "Write a high-performance C++ implementation of a thread pool."

# Apply chat template if available
if hasattr(tokenizer, "apply_chat_template"):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

inputs = tokenizer(
    prompt, 
    return_tensors="np",
    padding=False,
    truncation=True,
)

input_ids = jnp.array(inputs["input_ids"])
attention_mask = jnp.array(inputs.get("attention_mask", jnp.ones_like(input_ids)))

print(f"\nPrompt: {prompt}")
print(f"Input shape: {input_ids.shape}")

# ----------------------------------------------------------------------
# 6. GENERATION LOOP
# ----------------------------------------------------------------------
print("\nGenerating response...")

# Initialize RNG
rng = GenerateRNG(seed=42)

# Prepare generation config
generation_config = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

# Use model's generate method if available
if hasattr(model, 'generate'):
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        params=params,
        generation_config=generation_config,
        rng=rng.rng,
    ).sequences
else:
    # Fallback: manual generation loop
    print("Using manual generation loop...")
    
    current_ids = input_ids
    
    for step in range(MAX_NEW_TOKENS):
        # Forward pass
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )
        
        # Get logits for next token
        next_token_logits = outputs.logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / generation_config["temperature"]
        
        # Sample next token
        next_token = jax.random.categorical(
            rng.rng, 
            next_token_logits, 
            axis=-1
        )
        rng = GenerateRNG(rng=jax.random.split(rng.rng)[0])
        
        # Append to sequence
        current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
        attention_mask = jnp.ones_like(current_ids)
        
        # Check for EOS
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    output_ids = current_ids

# ----------------------------------------------------------------------
# 7. DECODE AND PRINT OUTPUT
# ----------------------------------------------------------------------
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("-" * 80)
print("GENERATED OUTPUT:")
print("-" * 80)
print(generated_text)
print("-" * 80)

print(f"\nGeneration complete. Generated {len(output_ids[0]) - len(input_ids[0])} tokens.")