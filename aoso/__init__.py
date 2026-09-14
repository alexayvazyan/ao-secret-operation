import os

# torch 2.14 routes some eager ops (e.g. RoPE) to Triton kernels that need a C compiler,
# which this WSL install lacks. Must be set before torch is first imported.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
