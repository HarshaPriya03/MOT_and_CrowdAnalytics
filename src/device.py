"""
Hardware acceleration detection with graceful fallback:
CUDA -> MPS (Apple Silicon) -> CPU.
"""
import torch


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"