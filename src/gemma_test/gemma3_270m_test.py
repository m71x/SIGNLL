import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer (setup time not included)
MODEL_NAME = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
model.eval()

# Input prompt
prompt = "What is 2+2?"

# ---------------------------
# START INFERENCE TIMING
# ---------------------------
start_time = time.time()

# Tokenize and move to device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=80)

# Synchronize GPU to ensure timing is accurate
if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()
# ---------------------------
# END INFERENCE TIMING
# ---------------------------

# Decode result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nGenerated Text:\n{generated_text}")
print(f"\n⏱️ Inference time: {end_time - start_time:.4f} seconds")
