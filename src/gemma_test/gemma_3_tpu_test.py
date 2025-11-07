import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, GenerationConfig, GemmaTokenizer
import os
import subprocess

# TPU device
device = xm.torch_xla.device()  
print("Using device:", device, flush=True)

# --- GCS model location ---
gcs_path = "gs://startup-scripts-gemma3/gemma3-4b/gemma-3-4b-it"
local_model_path = "/home/mikexi/gemma_model/gemma-3-4b-it"

# --- Download model if missing ---
if not os.path.exists(local_model_path):
    print(f"Model not found locally. Downloading from {gcs_path}...", flush=True)
    os.makedirs(local_model_path, exist_ok=True)
    result = subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{gcs_path}/*", local_model_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error downloading model:", result.stderr, flush=True)
        raise SystemExit(1)
    print("Download complete.", flush=True)
else:
    print("Model already present locally.", flush=True)

# --- Load tokenizer ---
vocab_file = os.path.join(local_model_path, "tokenizer.model")
if not os.path.exists(vocab_file):
    raise ValueError(f"tokenizer.model not found in {local_model_path}")

tokenizer = GemmaTokenizer(vocab_file=vocab_file)
print("Tokenizer loaded.", flush=True)

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)
model.to(device)
print("Model loaded successfully.", flush=True)

# --- Example input ---
prompt = "In the future, artificial intelligence will"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generation configuration
gen_config = GenerationConfig(max_new_tokens=30, use_cache=False)

# --- Run inference ---
print("Running inference on TPU...", flush=True)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=False
    )

# Decode and print output
generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

print("\n--- Model Output ---", flush=True)
print(generated_text, flush=True)
