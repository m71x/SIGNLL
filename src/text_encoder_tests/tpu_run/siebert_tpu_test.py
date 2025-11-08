import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage

# --- GCS + Local paths ---
BUCKET_NAME = "encoder-models"   # change if your bucket name differs
GCS_PREFIX = "siebert"    # your Siebert model path in GCS
LOCAL_MODEL_PATH = "/home/mikexi/siebert_model"

# --- Function: download model from GCS ---
def download_model_from_gcs(bucket_name, prefix, local_dir):
    print(f"[Core 0] Downloading model from gs://{bucket_name}/{prefix} ...", flush=True)
    os.makedirs(local_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        local_file = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        count += 1
        print(f"[Core 0] Downloaded {count}: {rel_path}", flush=True)
    print("[Core 0] Model download complete!", flush=True)

# --- Main process per TPU core ---
def _mp_fn(index):
    device = xm.torch_xla.device()
    print(f"[Core {index}] Using device: {device}", flush=True)

    # Only core 0 downloads the model
    if xm.is_master_ordinal():
        if not os.path.exists(LOCAL_MODEL_PATH):
            download_model_from_gcs(BUCKET_NAME, GCS_PREFIX, LOCAL_MODEL_PATH)
        else:
            print("[Core 0] Model already exists locally.", flush=True)

    # Sync all workers before loading
    xm.rendezvous("model_download_done")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    if xm.is_master_ordinal():
        print("[Core 0] Tokenizer loaded.", flush=True)

    # --- Load model ---
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    if xm.is_master_ordinal():
        print("[Core 0] Model loaded successfully.", flush=True)

    # --- Run sample inference on all workers ---
    sample_text = "I really love this product. It works great and exceeded my expectations!"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    print(f"[Core {index}] Classification result: {predicted_class} (probabilities: {probabilities.tolist()})", flush=True)

    xm.rendezvous("done")

# --- Entry point ---
if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())
