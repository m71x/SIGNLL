"""Minimal test: load model and do one forward pass with hidden states."""
import jax
import jax.numpy as jnp
jax.distributed.initialize()
from jax.experimental import multihost_utils
import easydel as ed
import inspect

is_master = jax.process_index() == 0
if is_master:
    print(f"Devices: {jax.device_count()}, Local: {jax.local_device_count()}")

MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
axis_dims = (1, 1, 8, 4, 1)
axis_names = ("dp", "fsdp", "tp", "sp", "selective")

if is_master:
    print("Loading model...")

elm = (
    ed.eLargeModel.from_pretrained(MODEL_ID)
    .set_dtype("bf16")
    .set_sharding(axis_dims=axis_dims, axis_names=axis_names)
    .set_esurge(max_model_len=4096, max_num_seqs=4)
)

if is_master:
    print("Building eSurge to init model...")
esurge = elm.build_esurge()

if is_master:
    print("Tearing down eSurge...")
del esurge
import gc
gc.collect()
jax.clear_caches()
gc.collect()

multihost_utils.sync_global_devices("cleanup")

# Monkey-patch output validation
import easydel.infra.modeling_outputs as _mo
for _name, _cls in inspect.getmembers(_mo, inspect.isclass):
    if hasattr(_cls, '__post_init__'):
        _cls.__post_init__ = lambda self: None

if is_master:
    print("Attempting forward pass...")

model_mesh = elm._model.config.mesh
test_ids = jnp.ones((1, 10), dtype=jnp.int32)

with model_mesh:
    out = elm._model(input_ids=test_ids, output_hidden_states=True)

if is_master:
    print(f"SUCCESS! Logits shape: {out.logits.shape}")
    print(f"Hidden states: {len(out.hidden_states)} layers")
    print("Forward pass works!")

multihost_utils.sync_global_devices("done")
