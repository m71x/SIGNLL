import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# 1. Load pretrained TinyBERT SST-2 distilled model + tokenizer
# -----------------------------------------------------------

# The fine-tuned SST-2 version:
# (If this doesn't exist locally, you can fine-tune manually or check for "tiny-bert-sst2-distilled")
# For demonstration, we'll use TinyBERT and simulate the SST2 head
# but you can replace with: "tiny-bert-sst2-distilled" if it’s on HF

if torch.cuda.is_available():
    print("GPU is available.")
    # You can also get more information about the GPU
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. PyTorch will use the CPU.")

model_name = "Alireza1044/mobilebert_sst2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

# -----------------------------------------------------------
# 2. Move model to GPU if available
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------------------------------------
# 3. Prepare example text input
# -----------------------------------------------------------
text = "CS128 is a good class"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # move to GPU

# -----------------------------------------------------------
# 4. Forward pass with hidden states enabled
# -----------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# -----------------------------------------------------------
# 5. Extract and print hidden states
# -----------------------------------------------------------
hidden_states = outputs.hidden_states  # tuple: (embeddings + each transformer layer output)

print(f"\nNumber of hidden layers (including embeddings): {len(hidden_states)}")
print("Shapes of each hidden layer:")
for i, h in enumerate(hidden_states):
    print(f"Layer {i}: {h.shape}")

# Example: extract [CLS] token representations across layers
cls_reps = [h[:, 0, :].squeeze().cpu().numpy() for h in hidden_states]

# -----------------------------------------------------------
# 6. Display final sentiment prediction
# -----------------------------------------------------------
logits = outputs.logits
pred = torch.argmax(logits, dim=-1).item()
labels = ["negative", "positive"]

print(f"\nInput text: {text}")
print(f"Predicted sentiment: {labels[pred]}")
print(f"Raw logits: {logits.cpu().numpy()}")

# -----------------------------------------------------------
# 7. (Optional) visualize or inspect representations
# -----------------------------------------------------------
# For example: print first 5 dimensions of CLS vector from each layer
print("\n[CLS] token first 5 dims for each layer:")
for i, rep in enumerate(cls_reps):
    print(f"Layer {i}: {rep[:5]}")

