from __future__ import annotations

from typing import Dict, List

G1_PRESETS: List[Dict[str, str]] = [
    {
        "id": "g1-0.1b",
        "label": "RWKV-7 G1 0.1B",
        "size": "0.1B",
        "best_for": "testing / very limited hardware",
        "family": "rwkv7-g1",
        "official": True,
    },
    {
        "id": "g1-0.4b",
        "label": "RWKV-7 G1 0.4B",
        "size": "0.4B",
        "best_for": "lightweight CPUs and laptops",
        "family": "rwkv7-g1",
        "official": True,
    },
    {
        "id": "g1-1.5b",
        "label": "RWKV-7 G1 1.5B",
        "size": "1.5B",
        "best_for": "balanced local use",
        "family": "rwkv7-g1",
        "official": True,
    },
    {
        "id": "g1-2.9b",
        "label": "RWKV-7 G1 2.9B",
        "size": "2.9B",
        "best_for": "faster GPU-backed local use",
        "family": "rwkv7-g1",
        "official": True,
    },
    {
        "id": "g1-7.2b",
        "label": "RWKV-7 G1 7.2B",
        "size": "7.2B",
        "best_for": "high quality on strong hardware",
        "family": "rwkv7-g1",
        "official": True,
    },
]

def recommend_preset(vram_gb: int | None = None) -> Dict[str, str]:
    if vram_gb is None:
        return G1_PRESETS[2]
    if vram_gb >= 16:
        return G1_PRESETS[4]
    if vram_gb >= 8:
        return G1_PRESETS[3]
    if vram_gb >= 4:
        return G1_PRESETS[2]
    if vram_gb >= 2:
        return G1_PRESETS[1]
    return G1_PRESETS[0]
