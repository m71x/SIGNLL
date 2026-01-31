import jax
import sys
import os
import gc # Added for memory cleanup

# 1. CRITICAL: Initialize distributed system BEFORE importing EasyDeL
jax.distributed.initialize() 
from jax.experimental import multihost_utils

# 2. NOW it is safe to import EasyDeL
import easydel as ed

# Determine if this specific VM is the primary worker
is_master = jax.process_index() == 0

if is_master:
    print(f"Total devices: {jax.device_count()}")
    print(f"Local devices: {jax.local_device_count()}")
    print("Starting model initialization...")


