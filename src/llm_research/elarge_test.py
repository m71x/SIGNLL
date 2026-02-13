import jax
import jax.numpy as jnp
import numpy as np
import sys
import gc
import json
import inspect

# 1. Initialize distributed system
jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
from transformers import AutoTokenizer

is_master = jax.process_index() == 0

# ── PROMPTS ─────────────────────────────────────────────────────────────
PROMPTS = [
    # --- Coding / Programming ---
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a Python function that checks whether a string is a valid palindrome, ignoring spaces and punctuation.",
    "What is the time complexity of merge sort and why?",
    "Explain what a closure is in JavaScript with a short example.",
    "Write a SQL query to find the second highest salary from an employees table.",
    "What are the SOLID principles in object-oriented design? Summarize each in one sentence.",
    "Explain the difference between a stack and a queue with real-world analogies.",
    "Write a Python generator that yields the Fibonacci sequence indefinitely.",
    "What is the difference between concurrency and parallelism?",
    "Explain how garbage collection works in Java.",
    # --- Math / Logic ---
    "Prove that the square root of 2 is irrational.",
    "What is the difference between permutations and combinations? Give an example of each.",
    "Explain Bayes' theorem in simple terms with an everyday example.",
    "What is the derivative of x^x and how do you compute it?",
    "Solve the equation 2x + 5 = 3x - 7 and explain each step.",
    "What is the fundamental theorem of calculus in plain English?",
    "Explain the pigeonhole principle and give a fun application of it.",
    "What is modular arithmetic? Explain with clock-based examples.",
    "Describe the difference between a discrete and continuous probability distribution.",
    "Explain what eigenvalues and eigenvectors mean geometrically.",
    # --- Science / Technical ---
    "How does CRISPR gene editing work at a high level?",
    "Explain the theory of general relativity in terms a high schooler could understand.",
    "What causes the northern lights (aurora borealis)?",
    "Describe how mRNA vaccines work step by step.",
    "What is quantum entanglement and why is it considered 'spooky'?",
    "Explain the greenhouse effect and its relationship to climate change.",
    "How do neural networks learn? Explain backpropagation simply.",
    "What is the difference between nuclear fission and nuclear fusion?",
    "Explain the Doppler effect with examples from sound and light.",
    "What is entropy in thermodynamics versus information theory?",
    # --- Creative / Writing ---
    "Write a haiku about a rainy Monday morning.",
    "Write a short horror story in exactly 100 words.",
    "Compose a limerick about a programmer who loves debugging.",
    "Write a persuasive paragraph arguing that pineapple belongs on pizza.",
    "Describe a sunset over the ocean using only metaphors.",
    "Write a formal email politely declining a meeting invitation.",
    "Create a dialogue between a cat and a dog debating who is the better pet.",
    "Write the opening paragraph of a mystery novel set in 1920s Paris.",
    "Compose a short motivational speech for someone starting a new job.",
    "Write a product description for a fictional self-tying shoelace.",
    # --- General Knowledge / Reasoning ---
    "What were the main causes of World War I?",
    "Explain the trolley problem and the main ethical perspectives on it.",
    "Summarize the plot of George Orwell's 1984 in one paragraph.",
    "What is the difference between weather and climate?",
    "Explain how the electoral college works in the United States.",
    "What is the overview effect that astronauts experience?",
    "Describe the water cycle in detail.",
    "What is the Turing test and has any AI passed it?",
    "Explain the concept of opportunity cost with a practical example.",
    "Why do we have leap years? Explain the calendar mathematics behind it.",
]

if is_master:
    print(f"Starting multi-prompt test with {len(PROMPTS)} prompts")
    print("=" * 70)

# 2. Load tokenizer and model
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

elm = (
    ed.eLargeModel.from_pretrained(model_id)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

# Monkey-patch EasyDeL output dataclass validation for multi-host (once).
# In eager mode on multi-host, intermediate arrays are sharded across hosts
# and fail is_fully_replicated/is_fully_addressable checks in __post_init__.
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

if is_master:
    print("Patched output validation for multi-host forward pass")

# Build esurge once to populate elm._model, then grab the mesh
esurge = elm.build_esurge()
model_mesh = elm._model.config.mesh
del esurge
gc.collect()
multihost_utils.sync_global_devices("initial_esurge_cleanup")

# ── Collect results across all prompts ──────────────────────────────────
all_results = []

for prompt_idx, prompt in enumerate(PROMPTS):
    if is_master:
        print(f"\n{'=' * 70}")
        print(f"[{prompt_idx + 1}/{len(PROMPTS)}] PROMPT: {prompt}")
        print(f"{'=' * 70}")

    # ── STEP 1: GENERATION ──────────────────────────────────────────────
    esurge = elm.build_esurge()
    conversation = [{"role": "user", "content": prompt}]

    if is_master:
        print("Response: ", end="", flush=True)

    # IMPORTANT: Accumulate generated_text on ALL workers, not just master.
    # All workers must have identical text for the forward pass, otherwise
    # tokenization produces different sequences on each worker → TPU halt.
    generated_text = ""
    for output in esurge.chat(
        conversation,
        sampling_params=ed.SamplingParams(max_tokens=256),
        stream=True,
    ):
        generated_text += output.delta_text
        if is_master:
            print(output.delta_text, end="", flush=True)

    if is_master:
        print("\n")

    # ── STEP 2: ACTIVATION EXTRACTION (Forward Pass) ────────────────────
    if is_master:
        print("--- Shutting down eSurge before forward pass ---")

    del esurge
    gc.collect()
    multihost_utils.sync_global_devices(f"esurge_stopped_{prompt_idx}")

    full_conversation = conversation + [{"role": "assistant", "content": generated_text}]

    input_ids = tokenizer.apply_chat_template(
        full_conversation,
        return_tensors="np",
        add_generation_prompt=False,
    )

    if is_master:
        print(f"Tokenized sequence length: {input_ids.shape[1]}")

    # Forward pass inside the mesh context
    with model_mesh:
        if is_master:
            print("Running forward pass...")
        model_outputs = elm._model(
            input_ids=jnp.array(input_ids),
            output_hidden_states=True,
        )

    # Last layer activations  →  shape: (batch, seq_len, hidden_dim)
    last_layer_activations = model_outputs.hidden_states[-1]
    logits = model_outputs.logits

    # ALL workers must participate in sharded array ops
    num_layers = len(model_outputs.hidden_states)
    act = last_layer_activations[0]  # (seq_len, hidden_dim)
    mean_val = float(jnp.mean(act))
    std_val = float(jnp.std(act))
    min_val = float(jnp.min(act))
    max_val = float(jnp.max(act))

    probs = jax.nn.softmax(logits, axis=-1)
    seq_len = input_ids.shape[1]
    last_n = min(5, seq_len)

    # Pre-compute per-token data on ALL workers
    token_data = []
    for pos in range(seq_len - last_n, seq_len):
        token_id = int(input_ids[0, pos])
        top_k_indices = jnp.argsort(probs[0, pos, :])[-3:][::-1]
        top_k_probs = probs[0, pos, top_k_indices]
        act_slice = last_layer_activations[0, pos, :5]
        top_k_ids = [int(top_k_indices[k]) for k in range(3)]
        top_k_ps = [float(top_k_probs[k]) for k in range(3)]
        act_vals = act_slice.tolist()
        token_data.append((pos, token_id, top_k_ids, top_k_ps, act_vals))

    # Only master prints results
    if is_master:
        print(f"Number of hidden-state layers returned: {num_layers}")
        print(f"Last-layer activation shape: {last_layer_activations.shape}")

        print(f"\nLast-layer activation statistics:")
        print(f"  mean : {mean_val:.6f}")
        print(f"  std  : {std_val:.6f}")
        print(f"  min  : {min_val:.6f}")
        print(f"  max  : {max_val:.6f}")

        print(f"\nPer-token detail (last {last_n} positions):")
        for pos, token_id, top_k_ids, top_k_ps, act_vals in token_data:
            token_str = tokenizer.decode([token_id])
            print(f"\n  Position {pos} — token: '{token_str}'")
            print(f"    Activation (first 5 dims): {act_vals}")
            for k in range(3):
                t_name = tokenizer.decode([top_k_ids[k]])
                print(f"    Top-{k+1} prediction: '{t_name}' ({top_k_ps[k]*100:.2f}%)")

    # ── Collect result for this prompt ──────────────────────────────────
    result = {
        "prompt_index": prompt_idx,
        "prompt": prompt,
        "response": generated_text,
        "seq_len": int(seq_len),
        "num_layers": int(num_layers),
        "activation_stats": {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        },
        "token_details": [
            {
                "position": pos,
                "token_id": token_id,
                "token_str": tokenizer.decode([token_id]),
                "top_3_predictions": [
                    {
                        "token": tokenizer.decode([top_k_ids[k]]),
                        "probability": top_k_ps[k],
                    }
                    for k in range(3)
                ],
                "activation_first_5": act_vals,
            }
            for pos, token_id, top_k_ids, top_k_ps, act_vals in token_data
        ],
    }
    all_results.append(result)

    # Free activation memory before next iteration
    del model_outputs, last_layer_activations, logits, probs, act
    gc.collect()
    multihost_utils.sync_global_devices(f"prompt_done_{prompt_idx}")

    if is_master:
        print(f"\n✓ Prompt {prompt_idx + 1}/{len(PROMPTS)} complete")

# ── SAVE RESULTS ────────────────────────────────────────────────────────
if is_master:
    output_path = "elarge_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"All {len(PROMPTS)} prompts complete. Results saved to {output_path}")
    print(f"{'=' * 70}")

# ── CLEANUP ─────────────────────────────────────────────────────────────
if is_master:
    print("\nCleaning up...")
del elm
gc.collect()
multihost_utils.sync_global_devices("ready_to_kill")
jax.distributed.shutdown()
sys.exit(0)