import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

# 1. Detect your 32 chips (v4-64 has 32 chips / 64 TensorCores)
devices = jax.devices()
num_devices = len(devices) # Should be 32

# 2. Define a 2D Logical Mesh
# We use 4-way Data Parallelism and 8-way Model (Tensor) Parallelism
# This matches the 32 physical chips (4 * 8 = 32)
data_parallelism = 4
model_parallelism = 8

device_mesh = mesh_utils.create_device_mesh((data_parallelism, model_parallelism))
mesh = Mesh(device_mesh, axis_names=('data', 'model'))

# 3. Define Sharding Rules
# For model parameters: Shard the 'hidden' or 'output' dimension by 'model'
# For inputs/activations: Shard the 'batch' dimension by 'data'
param_sharding = NamedSharding(mesh, P(None, 'model'))
data_sharding = NamedSharding(mesh, P('data', None))

# 4. Apply sharding to your Qwen model (using your earlier Flax setup)
# This constraint ensures the model is distributed properly before execution
def apply_sharding(model_state):
    return jax.lax.with_sharding_constraint(model_state, param_sharding)

print(f"Mesh created: {mesh}")
print(f"Sharding model 32B across {num_devices} chips...")