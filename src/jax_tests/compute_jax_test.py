import jax
import jax.numpy as jnp
import time

# Get device info
global_device_count = jax.device_count()
local_device_count = jax.local_device_count()


def heavy_compute(x):
    """Perform a few large matrix multiplies + nonlinearities."""
    for _ in range(5):
        x = jnp.tanh(x @ x.T)  # O(n^3) matmul + activation
    return jnp.sum(x)  # single scalar per device


# Define the per-device computation function
def per_device_compute(seed):
    key = jax.random.PRNGKey(seed)
    # Make a moderately large matrix (~4096x4096 floats)
    mat = jax.random.normal(key, (4096, 4096))
    out = heavy_compute(mat)
    # Cross-reduce (sum across all devices)
    return jax.lax.psum(out, axis_name='i')


# Parallelize the function across TPU cores
mapped_heavy_compute = jax.pmap(per_device_compute, axis_name='i')

# Prepare one seed per local device
seeds = jnp.arange(local_device_count)

# Warm-up compile (first run triggers XLA compilation)
if jax.process_index() == 0:
    print(f"Compiling on {global_device_count} TPU cores...")

start_time = time.time()
results = mapped_heavy_compute(seeds).block_until_ready()
elapsed = time.time() - start_time

if jax.process_index() == 0:
    print(f"✅ Global device count: {global_device_count}")
    print(f"✅ Local device count:  {local_device_count}")
    print(f"✅ pmap result shape:   {results.shape}")
    print(f"✅ Computation time:    {elapsed:.2f} s")
    print(f"✅ Sample result:       {results[0]}")
