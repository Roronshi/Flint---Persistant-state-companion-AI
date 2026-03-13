from __future__ import annotations

import os
from typing import Callable, Optional

from .base_backend import BaseModelBackend


class DummyBackend(BaseModelBackend):
    backend_kind = "dummy"
    supports_reasoning_mode = False
    supports_persistent_state = True
    supports_adapter_training = False

    def __init__(self) -> None:
        self.state = None

    def load(self, path: str, strategy: str | None = None, vocab_path: str | None = None) -> None:
        return None

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.85,
        top_k: int = 0,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, int, float]:
        text = "[Dummy model] Detta är en platsmodell som används när RWKV-modellen inte är tillgänglig."
        if prompt:
            prefix = " ".join(prompt.strip().split()[:8])
            if prefix:
                text += f" Du skrev: {prefix}..."
        if stream_callback:
            stream_callback(text)
        return text, len(text.split()), 0.0

    def reset_state(self) -> None:
        self.state = None

    def save_state(self, path: str) -> None:
        if self.state is None:
            return
        try:
            import torch
            tmp = path + ".tmp"
            torch.save(self.state, tmp)
            os.replace(tmp, path)
        except ImportError:
            pass

    def load_state(self, path: str) -> bool:
        if os.path.exists(path):
            try:
                import torch
                self.state = torch.load(path, map_location="cpu", weights_only=False)
                return True
            except ImportError:
                pass
        self.state = None
        return False
