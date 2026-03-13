from .base_backend import BaseModelBackend
from .dummy_backend import DummyBackend

try:
    from .rwkv_backend import RWKVBackend
except ImportError:  # torch or rwkv not installed
    from .dummy_backend import DummyBackend as RWKVBackend  # type: ignore[misc]

try:
    from .onnx_backend import ONNXBackend
except ImportError:
    from .dummy_backend import DummyBackend as ONNXBackend  # type: ignore[misc]

__all__ = ["BaseModelBackend", "DummyBackend", "RWKVBackend", "ONNXBackend"]
