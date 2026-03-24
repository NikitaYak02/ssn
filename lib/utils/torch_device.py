"""
Helpers for selecting and synchronizing PyTorch devices.
"""


def get_torch_device(torch_module) -> str:
    """
    Prefer CUDA, then Apple Metal (MPS), then CPU.
    """
    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def synchronize_device(torch_module, device: str) -> None:
    """
    Synchronize the active accelerator stream when supported.
    """
    if device == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        return

    if device == "mps":
        mps_module = getattr(torch_module, "mps", None)
        if mps_module is not None and hasattr(mps_module, "synchronize"):
            mps_module.synchronize()
