import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model: RoBERTa sentiment classifier (3 labels)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load tokenizer and model (setup time not counted)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Example input
text = "I absolutely loved this movie, it was fantastic! It was pretty cool to see for the first time. The visual effects were amazing, and I didn't notice that the movie also won an Oscar at first. However, it is so great!"

# ---------------------------
# START INFERENCE TIMING
# ---------------------------
start_time = time.time()

# Tokenize + move to GPU
inputs = tokenizer(text, return_tensors="pt").to(device)

# Forward pass (get logits and hidden states)
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    logits = outputs.logits

end_time = time.time()
# ---------------------------
# END INFERENCE TIMING
# ---------------------------

# Predicted sentiment
predicted_class = torch.argmax(logits, dim=-1).item()
labels = ["negative", "neutral", "positive"]

print(f"Predicted label: {labels[predicted_class]}")

# Extract <s> (class) token embeddings at each layer
cls_states = [layer[:, 0, :].squeeze(0).cpu() for layer in hidden_states]

print(f"\nNumber of layers (including embedding): {len(cls_states)}")
print(f"CLS token embedding shape: {cls_states[0].shape}")

# Print first few layer stats
for i, state in enumerate(cls_states[:12]):
    print(f"Layer {i} CLS vector norm: {state.norm().item():.4f}")

# Print inference time
print(f"\n⏱️ Inference time: {end_time - start_time:.4f} seconds")
