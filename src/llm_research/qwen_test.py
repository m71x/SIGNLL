import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# =========================================================================
# CONFIGURATION
# =========================================================================
FLAGS = {
    # We can try the 32B model now since we have 200GB+ RAM space in /dev/shm
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "prompts": [
        "write a python function to merge two sorted lists.",
        "explain the difference between tcp and udp.",
        "write a c++ class for a binary search tree.",
        "how do i deploy a docker container to kubernetes?",
        "write a rust function to check for palindromes.",
        "explain the concept of dependency injection.",
        "write a sql query to find the second highest salary.",
        "create a simple react component for a login form."
    ]
}

def run_inference(rank, flags):
    # 1. Initialize Device
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    
    # 2. Set Cache to /dev/shm (RAM Disk) to bypass full disk
    # This gives us ~200GB of space for the download
    ram_cache_dir = "/dev/shm/huggingface_cache"
    os.makedirs(ram_cache_dir, exist_ok=True)
    
    # 3. Load Tokenizer
    # We download to our custom RAM path
    tokenizer = AutoTokenizer.from_pretrained(
        flags["model_id"], 
        trust_remote_code=True, 
        cache_dir=ram_cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Model directly to TPU
    if rank == 0:
        print(f"Downloading {flags['model_id']} to RAM ({ram_cache_dir})...")
    
    # Load model with cache_dir pointed to RAM
    model = AutoModelForCausalLM.from_pretrained(
        flags["model_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=ram_cache_dir
    ).to(device)
    
    # 5. Distribute Prompts
    my_prompts = flags["prompts"][rank % len(flags["prompts"])]
    
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful coding assistant."},
        {"role": "user", "content": my_prompts}
    ]
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 6. Tokenize & Move to TPU
    inputs = tokenizer(
        text_input, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if rank == 0:
        print("Starting generation...")

    # 7. Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=flags["max_new_tokens"],
            temperature=flags["temperature"],
            top_p=flags["top_p"],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 8. Decode and Print
    generated_ids_cpu = generated_ids.cpu()
    input_len = input_ids.shape[1]
    response_tokens = generated_ids_cpu[:, input_len:]
    response_text = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)[0]

    # Print results safely
    for i in range(world_size):
        xm.rendezvous(f"print_turn_{i}")
        if rank == i:
            print(f"\n{'='*40}")
            print(f"DEVICE {rank} | Prompt: {my_prompts}")
            print(f"{'-'*40}")
            print(f"RESPONSE:\n{response_text}")
            print(f"{'='*40}\n")
        xm.mark_step()

    if rank == 0:
        print("Inference Complete.")

def _mp_fn(rank, flags):
    os.environ["PJRT_DEVICE"] = "TPU"
    run_inference(rank, flags)

if __name__ == "__main__":
    print("Launching Qwen2.5-Coder (Using /dev/shm for storage)...")
    xmp.spawn(_mp_fn, args=(FLAGS,), start_method="fork")