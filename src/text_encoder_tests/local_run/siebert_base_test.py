import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "siebert/sentiment-roberta-large-english"

# Load tokenizer and model (setup not timed)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,    # ensure hidden states are returned
)

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Example input
text = "I hate"

# ------------- Start timing -------------
start_time = time.time()

# Tokenize and move to device
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

# Forward pass (get logits + hidden states)
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple: (embeddings layer output, then each transformer layer)
    logits = outputs.logits

# If using GPU, ensure all ops complete for accurate timing
if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()
# ------------- End timing -------------

# Convert logits to predicted label
predicted_idx = torch.argmax(logits, dim=-1).item()
label_map = {0: "negative", 1: "positive"}    # model card says binary 0/1 for neg/pos. :contentReference[oaicite:2]{index=2}
predicted_label = label_map.get(predicted_idx, str(predicted_idx))

print(f"Predicted label: {predicted_label}")

# Extract class-token (first token) state from each layer
cls_states = [ layer[:, 0, :].squeeze(0).cpu() for layer in hidden_states ]
print(f"\nNumber of hidden states layers (incl. embeddings): {len(cls_states)}")
print(f"CLS token vector shape (for layer 0): {cls_states[0].shape}")

# Example: print norm of the class token vector for the first few layers
for i, vec in enumerate(cls_states[:24]):
    print(f"Layer {i} CLS vector norm: {vec.norm().item():.4f}")

# Print inference time
print(f"\n⏱ Inference time (tokenize + forward): {end_time - start_time:.4f} seconds")
