from __future__ import annotations

TRAINING_PRESETS = [
    {
        "id": "companion_safe",
        "label": "Companion Safe",
        "method": "LoRA",
        "mode": "beta-safe",
        "note": "Stable standard mode for local companion fine-tuning.",
    },
    {
        "id": "low_vram_lab",
        "label": "Low VRAM Lab",
        "method": "QLoRA",
        "mode": "future-lab",
        "note": "Planned low-memory path. Experimental and not enabled yet.",
    },
    {
        "id": "state_memory_lab",
        "label": "State Memory Lab",
        "method": "State Tuning",
        "mode": "future-lab",
        "note": "Planned state-centric adaptation path. Experimental and not enabled yet.",
    },
]

def default_training_preset():
    return TRAINING_PRESETS[0]
