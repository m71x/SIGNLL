from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_xla.core.xla_model as xm
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", torch_dtype=torch.bfloat16).to(xm.torch_xla.device())

prompt = "Why are TPUs efficient for deep learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(xm.torch_xla.device())

outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
