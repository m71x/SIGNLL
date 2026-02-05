import jax
import jax.numpy as jnp
import sys
import os
import gc

jax.distributed.initialize() 
from jax.experimental import multihost_utils
import easydel as ed

# 1. Initialize distributed system

from jax.experimental import multihost_utils

# 2. Setup model and engine
is_master = jax.process_index() == 0

if is_master:
    print("Starting model initialization...")

elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
    .set_dtype("bf16")
    .set_sharding(
        axis_dims=(1, 1, 8, 4, 1), 
        axis_names=("dp", "fsdp", "tp", "sp", "selective")
    )
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

esurge = elm.build_esurge()
prompts = ["Explain the difference between TCP and UDP in one paragraph."]

for i, user_prompt in enumerate(prompts):
    if is_master:
        print(f"\nPROMPT: {user_prompt}\nResponse: ", end="", flush=True)

    conversation = [{"role": "user", "content": user_prompt}]
    
    # --- STEP 1: GENERATION ---
    # We collect deltas so we can reconstruct the full string for the forward pass
    generated_text = ""
    for output in esurge.chat(
        conversation,
        sampling_params=ed.SamplingParams(max_tokens=256),
        stream=True,
    ):
        if is_master:
            print(output.delta_text, end="", flush=True)
            generated_text += output.delta_text

    # --- STEP 2: OBSERVATION (Forward Pass) ---
    if is_master:
        print("\n\n--- Starting Activation Extraction ---")

    # Reconstruct the full conversation context
    full_convo = conversation + [{"role": "assistant", "content": generated_text}]
    
    # Tokenize the full conversation
    input_ids = elm.tokenizer.apply_chat_template(
        full_convo, 
        return_tensors="np", 
        add_generation_prompt=False
    )

    # Perform a manual forward pass to get hidden states and logits
    # We use elm.model.apply, which is the functional core of the model
    model_outputs = elm.model.apply(
        {'params': elm.params},
        input_ids=input_ids,
        output_hidden_states=True, # This captures all layers
        return_dict=True,
        train=False
    )

    # Extract final layer activations (Shape: [Batch, Seq_Len, Hidden_Dim])
    final_layer = model_outputs.hidden_states[-1]
    
    # Calculate probabilities from logits for the "Observation" analysis
    probs = jax.nn.softmax(model_outputs.logits, axis=-1)

    if is_master:
        # Example: Show data for the last 3 generated tokens
        last_n = 3
        seq_len = input_ids.shape[1]
        
        print(f"Total Sequence Length: {seq_len}")
        print(f"Final Layer Activation Shape: {final_layer.shape}")

        for pos in range(seq_len - last_n, seq_len):
            token_id = input_ids[0, pos]
            token_str = elm.tokenizer.decode([token_id])
            
            # Get Top 3 most likely tokens at this position for analysis
            top_k_indices = jnp.argsort(probs[0, pos, :])[-3:][::-1]
            top_k_probs = probs[0, pos, top_k_indices]
            
            print(f"\nToken at pos {pos}: '{token_str}'")
            print(f" - Activation Vector (first 5 dims): {final_layer[0, pos, :5]}")
            for k in range(3):
                t_name = elm.tokenizer.decode([top_k_indices[k]])
                print(f" - Prob {k+1}: {t_name} ({top_k_probs[k]*100:.2f}%)")

# Graceful Exit
if is_master: print("\nCleaning up...")
del esurge, elm
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)