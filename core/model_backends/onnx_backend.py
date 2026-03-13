from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import config
from .base_backend import BaseModelBackend

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


def _np_dtype_from_onnx_type(type_str: str):
    t = (type_str or "").lower()
    if "int64" in t:
        return np.int64
    if "int32" in t:
        return np.int32
    if "int16" in t:
        return np.int16
    if "float16" in t:
        return np.float16
    if "float64" in t:
        return np.float64
    if "bool" in t:
        return np.bool_
    return np.float32


def _safe_shape(shape):
    out = []
    for dim in shape or []:
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        else:
            out.append(1)
    return tuple(out) if out else (1,)


class ByteTokenizer:
    """Very small fallback tokenizer.

    This is not RWKV-ideal, but it allows deterministic ONNX graph exercising
    when the normal tokenizer stack is unavailable in the local environment.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        cleaned = [max(0, min(255, int(t))) for t in tokens]
        return bytes(cleaned).decode("utf-8", errors="ignore")


@dataclass
class TensorSpec:
    name: str
    shape: tuple
    dtype: object


class ONNXBackend(BaseModelBackend):
    backend_kind = "onnx"
    supports_reasoning_mode = True
    supports_persistent_state = True
    supports_adapter_training = False

    def __init__(self) -> None:
        self.session = None
        self.state = None
        self.path = None
        self.input_names = []
        self.output_names = []
        self.tokenizer = ByteTokenizer()
        self._fallback_reason = "Using byte tokenizer fallback"
        self._stop_event = threading.Event()

        self.token_input_name = None
        self.state_input_specs: list[TensorSpec] = []
        self.logits_output_name = None
        self.state_output_names: list[str] = []
        self.graph_ready = False

    def load(self, path: str, strategy: str | None = None, vocab_path: str | None = None) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime package is not installed")
        self.path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._classify_graph()

    def _classify_graph(self) -> None:
        if self.session is None:
            self.graph_ready = False
            return

        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())

        def is_token_name(name: str) -> bool:
            n = name.lower()
            return any(k in n for k in ["token", "tokens", "input_ids", "idx"])

        token_input = next((i for i in inputs if is_token_name(i.name)), None)
        if token_input is None and inputs:
            token_input = inputs[0]

        self.token_input_name = token_input.name if token_input else None
        self.state_input_specs = []
        for i in inputs:
            if i.name == self.token_input_name:
                continue
            self.state_input_specs.append(
                TensorSpec(
                    name=i.name,
                    shape=_safe_shape(getattr(i, "shape", None)),
                    dtype=_np_dtype_from_onnx_type(getattr(i, "type", "tensor(float)")),
                )
            )

        def is_logits_name(name: str) -> bool:
            n = name.lower()
            return any(k in n for k in ["logits", "logit", "output", "out"])

        logits_output = next((o for o in outputs if is_logits_name(o.name)), None)
        if logits_output is None and outputs:
            logits_output = outputs[0]

        self.logits_output_name = logits_output.name if logits_output else None
        self.state_output_names = [o.name for o in outputs if o.name != self.logits_output_name]
        self.graph_ready = bool(self.token_input_name and self.logits_output_name)

    def _graph_signature(self) -> str:
        if not self.session:
            return "unloaded"
        return f"inputs={self.input_names}, outputs={self.output_names}, token={self.token_input_name}, logits={self.logits_output_name}"

    def _ensure_state(self):
        if self.state is None:
            self.state = []
            for spec in self.state_input_specs:
                self.state.append(np.zeros(spec.shape, dtype=spec.dtype))
        return self.state

    def _prepare_token_array(self, tokens: list[int]):
        if self.session is None or self.token_input_name is None:
            raise RuntimeError("ONNX graph is not initialised")
        input_meta = next((i for i in self.session.get_inputs() if i.name == self.token_input_name), None)
        rank = len(getattr(input_meta, "shape", []) or [])
        arr = np.array(tokens, dtype=np.int64)
        if rank >= 2:
            arr = arr.reshape(1, -1)
        return arr

    def _extract_logits(self, raw):
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr.astype(np.float64)
        if arr.ndim == 2:
            return arr[-1].astype(np.float64)
        # Typical case: [batch, seq, vocab]
        return arr[0, -1].astype(np.float64)

    def _sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        logits = np.asarray(logits, dtype=np.float64).copy()
        if temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / max(temperature, 1e-6)

        if top_k and top_k > 0 and top_k < logits.shape[-1]:
            idx = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[idx] = logits[idx]
            logits = mask

        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.clip(np.sum(probs), 1e-12, None)

        if 0 < top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumulative = np.cumsum(sorted_probs)
            keep = cumulative <= top_p
            if not np.any(keep):
                keep[0] = True
            filtered = np.zeros_like(probs)
            filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
            total = filtered.sum()
            if total > 0:
                probs = filtered / total

        return int(np.random.choice(len(probs), p=probs))

    def _run_graph(self, tokens: list[int]):
        if self.session is None or not self.graph_ready:
            raise RuntimeError("ONNX graph is not ready")
        feeds = {self.token_input_name: self._prepare_token_array(tokens)}
        states = self._ensure_state()
        for spec, value in zip(self.state_input_specs, states):
            feeds[spec.name] = value
        outputs = self.session.run(None, feeds)

        output_map = {}
        for name, value in zip(self.output_names, outputs):
            output_map[name] = value

        logits = self._extract_logits(output_map[self.logits_output_name])
        new_state = []
        for name in self.state_output_names:
            new_state.append(output_map[name])
        if self.state_input_specs and new_state:
            self.state = new_state
        return logits

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
        t0 = time.perf_counter()
        if self.session is None:
            raise RuntimeError("ONNX backend not loaded")
        if not self.graph_ready:
            text = (
                "[ONNX backend loaded] Flint can inspect this ONNX model but could not identify a token/logits graph path. "
                f"Graph signature: {self._graph_signature()}."
            )
            if stream_callback:
                stream_callback(text)
            elapsed = max(time.perf_counter() - t0, 1e-6)
            return text, max(1, len(text.split())), elapsed

        self._stop_event.clear()
        prompt_tokens = self.tokenizer.encode(prompt or " ")
        logits = None
        for tok in prompt_tokens:
            logits = self._run_graph([tok])

        if logits is None:
            logits = self._run_graph([32])  # space token fallback

        generated: list[int] = []
        occurrence: dict[int, int] = {}
        out_last = 0
        out_str = ""

        for _ in range(max_tokens):
            if self._stop_event.is_set():
                break

            adjusted = logits.copy()
            for uid, cnt in occurrence.items():
                if 0 <= uid < adjusted.shape[-1]:
                    adjusted[uid] -= config.ALPHA_PRESENCE + cnt * config.ALPHA_FREQUENCY
            if adjusted.shape[-1] > 0:
                adjusted[0] = -float("inf")

            token = self._sample_logits(adjusted, temperature=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
            generated.append(token)
            occurrence[token] = occurrence.get(token, 0) + 1

            decoded = self.tokenizer.decode(generated[out_last:])
            if decoded:
                if stream_callback:
                    stream_callback(decoded)
                out_str += decoded
                out_last = len(generated)

            stop_hit = False
            for stop_str in config.STOP_SEQUENCES:
                if stop_str and out_str.endswith(stop_str):
                    out_str = out_str[: -len(stop_str)]
                    stop_hit = True
                    break
            if stop_hit:
                break

            logits = self._run_graph([token])

        elapsed = max(time.perf_counter() - t0, 1e-6)
        return out_str, len(generated), elapsed

    def stop_generation(self) -> None:
        self._stop_event.set()

    def reset_state(self) -> None:
        self.state = None

    def save_state(self, path: str) -> None:
        tmp = path + ".tmp"
        payload: dict = {
            "state": self.state,
            "graph": self._graph_signature(),
            "token_input_name": self.token_input_name,
            "logits_output_name": self.logits_output_name,
            "state_output_names": self.state_output_names,
        }
        np.save(tmp, payload, allow_pickle=True)
        os.replace(tmp, path)

    def load_state(self, path: str) -> bool:
        if os.path.exists(path):
            try:
                blob = np.load(path, allow_pickle=True).item()
                if isinstance(blob, dict):
                    self.state = blob.get("state")
                else:
                    self.state = blob
            except Exception:
                # Fall back to torch if the file was saved by an older version
                try:
                    import torch
                    blob = torch.load(path, map_location="cpu", weights_only=False)
                    self.state = blob.get("state") if isinstance(blob, dict) else blob
                except Exception:
                    self.state = None
            return True
        self.state = None
        return False
