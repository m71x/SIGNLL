import torch_xla.core.xla_model as xm

devices = xm.get_xla_supported_devices()
print(f"Detected TPU devices: {devices}")
print(f"Total TPU cores: {len(devices)}")
