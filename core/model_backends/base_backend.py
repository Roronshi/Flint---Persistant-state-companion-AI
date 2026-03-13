from __future__ import annotations

from typing import Callable, Optional


class BaseModelBackend:
    backend_kind = "base"
    supports_reasoning_mode = False
    supports_persistent_state = False
    supports_adapter_training = False

    def load(self, path: str, strategy: str | None = None, vocab_path: str | None = None) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def encode_context(self, text: str) -> None:
        return None

    def stop_generation(self) -> None:
        return None

    def reset_state(self) -> None:
        return None

    def save_state(self, path: str) -> None:
        return None

    def load_state(self, path: str) -> bool:
        return False

    def load_lora(self, path: str) -> bool:
        return False
