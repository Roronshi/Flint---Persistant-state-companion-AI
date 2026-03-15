# Flint

Flint is a personal AI companion that runs entirely on your own hardware. It uses RWKV — a recurrent neural network architecture — which means it carries genuine internal state between sessions rather than reconstructing context from a text log each time. That state is saved to disk when you close the chat and restored when you open it again. On top of that, nightly LoRA fine-tuning shapes the model's character from your accumulated conversations over weeks and months.

No cloud, no subscription, no data leaving your machine. Everything — inference, memory, training — runs locally.

```bash
git clone https://github.com/Roronshi/Flint---Persistant-state-companion-AI/
cd Flint---Persistant-state-companion-AI
bash install.sh
bash start.sh
```

---

## How it works

Flint maintains memory across three layers operating on different timescales:

| Layer | What it is | Timescale |
|---|---|---|
| **RWKV State** | The model's live internal state — what is active in the relationship right now | Per session |
| **LoRA Adapter** | Character formation — fine-tuned weight deltas that accumulate your conversational history | Weeks, months |
| **SQLite log** | Verbatim conversation history, fully searchable, never deleted | Forever |

**RWKV state** is not a text summary or a retrieved memory — it is the model's actual recurrent state tensor, saved as a `.pt` file. When you start a session, the model loads that tensor and continues from it. This is categorically different from transformer-based chatbots that re-read a context window from scratch.

**LoRA training** runs automatically at 03:00 each night if enough new conversations have accumulated (default threshold: 3 sessions). It mixes recent sessions with a replay buffer of older ones to prevent new training from overwriting older character formation. The updated adapter is loaded at the next session start.

**Dream generation** runs during idle periods between conversations. The model generates autonomous inner monologue — opinions, curiosities, observations — using seed prompts designed to produce non-reactive thought rather than user-directed responses. These are stored as a separate reflection type and included in LoRA training data. The purpose is to prevent the model from collapsing into an echo of the user over time.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Roronshi/Flint---Persistant-state-companion-AI/
cd Flint---Persistant-state-companion-AI
bash install.sh
```

The installer creates a virtual environment, installs dependencies, and sets up data directories. No CUDA toolkit or system-level ML packages required — PyTorch with CUDA runtime is sufficient.

### 2. Get a model

Download an official **RWKV-7 G1** model from [HuggingFace (BlinkDL)](https://huggingface.co/BlinkDL) and place it in the `models/` directory. Alternatively, upload a model directly from the web UI after startup.

**Recommended sizes:**

| Size | VRAM | Notes |
|---|---|---|
| 0.4B | ~1 GB | CPU-capable, limited quality |
| 1.5B | ~3 GB | Good for lightweight GPU setups |
| 2.9B | ~6 GB | Solid mid-range option |
| **7.2B** | **~8 GB (INT8)** | **Recommended — best local quality** |

The 7.2B model with `cuda fp16i8` strategy fits comfortably on a 12 GB GPU and delivers strong conversational quality. Smaller models work on less hardware but produce noticeably simpler responses.

### 3. Configure

Copy `config.example.py` to `config_local.py` and set at minimum:

```python
MODEL_PATH     = "/path/to/models/rwkv7-g1e-7.2b-20260301-ctx8192.pth"
MODEL_STRATEGY = "cuda fp16i8"   # INT8 quantisation — ~8 GB VRAM

USER_NAME = "your_name"
BOT_NAME  = "Flint"
```

Everything else has sensible defaults.

### 4. Start

```bash
bash start.sh          # Starts backend and opens http://localhost:8000
bash start.sh --fg     # Foreground mode (logs to stdout)
bash stop.sh           # Stop background process
```

---

## Web UI

Dark, fast, no frills. Chat streams token by token. The sidebar gives you:

| Control | Function |
|---|---|
| **Save state** | Save the RWKV state to disk immediately (also auto-saves every 5 turns) |
| **Run LoRA now** | Trigger training immediately without waiting for the nightly schedule |
| **Reset state** | Clear the RWKV state and start the relationship fresh — log is preserved |
| **Search** | Full-text search across the entire conversation history |
| **Backup** | Create a timestamped zip of state, adapter, and database |

Accessible from mobile via [Tailscale](https://tailscale.com) — open `http://<tailscale-ip>:8000`.

---

## Configuration reference

All settings live in `config.py` with defaults. Override in `config_local.py` — that file is `.gitignore`'d and never committed.

```python
# Model
MODEL_PATH     = "models/rwkv7-g1e-7.2b-20260301-ctx8192.pth"
MODEL_STRATEGY = "cuda fp16i8"   # or "cpu fp32", or "cuda fp16 *20 -> cpu fp32" (split)

# Identity
USER_NAME = "user"
BOT_NAME  = "companion"

# LoRA
LORA_R          = 16       # Adapter rank — higher means more expressive but slower training
LORA_MIN_CONVOS = 3        # Minimum new sessions required to trigger nightly training
REPLAY_RATIO    = 0.3      # Fraction of training data drawn from older sessions
LORA_SCHEDULE   = "03:00"  # Nightly training time (24h)

# Dream (autonomous inner monologue)
DREAM_RATIO            = 0.25   # Fraction of training segments that are dreams
DREAM_MAX_TOKENS       = 150
DREAM_TEMPERATURE      = 0.88
DREAM_INTERVAL_SECONDS = 1800   # Generate dreams every 30 min of idle time

# Generation
MAX_TOKENS  = 500
TEMPERATURE = 1.0
TOP_P       = 0.85
```

**Split strategy** — if the model doesn't fit fully in VRAM, the rwkv package supports pushing layers to CPU RAM:

```python
MODEL_STRATEGY = "cuda fp16i8 *24 -> cpu fp32"
# First 24 layers on GPU (INT8), remaining layers on CPU RAM
```

This trades inference speed for VRAM headroom. LoRA training is unaffected — the trainer always offloads base weights to CPU RAM before training regardless of the inference strategy.

---

## What happens over time

The model doesn't change in any single conversation. The changes are slow and cumulative.

After enough conversations have accumulated (default: 3), the nightly training job runs. It takes recent sessions, mixes in a fraction of older ones, and trains the LoRA adapter matrices — the small sets of weight deltas applied on top of the frozen base model. The base model itself is never modified. The adapter gets incrementally closer to how you communicate, what you talk about, and how the relationship between you has developed.

Dream generation runs in the background between conversations. These are autonomous thoughts the model generates without any user input — opinions, observations, things it's curious about. They're included in training to counterbalance the natural tendency of a model trained only on conversations to become a mirror of whoever it's talking to.

The RWKV state carries the immediate texture of the relationship: the last things discussed, the current emotional register, unresolved threads. It is what makes the model feel continuous rather than episodic.

---

## Importing conversations

Feed in existing chat history as a starting point. Supported formats: ChatGPT export, Claude export.

```bash
# ChatGPT — Settings → Export data → conversations.json
python tools/parser.py --source chatgpt --file ~/Downloads/conversations.json

# Claude — Settings → Export
python tools/parser.py --source claude --file ~/Downloads/conversations.json

# Preview without writing
python tools/parser.py --source chatgpt --file export.json --dry-run
```

Imported conversations are included in the training data pool and count toward the LoRA threshold. The more history you import, the stronger the starting adapter.

---

## Backup

Flint automatically backs up the three files that constitute the full relationship state after each training run:

- RWKV state (`.pt`)
- LoRA adapter (`.pth`)
- SQLite database (`.db`)

Backups are stored under `data/backups/` and pruned to the 7 most recent. You can also trigger a backup manually from the UI sidebar or via `POST /api/backup/run`. Individual backups are downloadable as `.zip` via `GET /api/backup/{timestamp}/download`.

---

## Data layout

```
flint/
├── models/
│   └── rwkv7-g1e-7.2b-20260301-ctx8192.pth   # Base weights (download separately)
├── data/
│   ├── states/{USER_NAME}_state.pt             # RWKV state — active relationship memory
│   ├── lora_adapters/current_adapter.pth       # Personal adapter — trained character
│   ├── conversations.db                        # SQLite log — full history
│   └── backups/                                # Timestamped backup archives
└── config_local.py                             # Your local settings (gitignored)
```

To sync the same companion across multiple machines, replicate `data/states/` and `data/lora_adapters/` — e.g. via Syncthing. The base model only needs to live on the machine running inference.

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.12 |
| RAM | 16 GB | 32 GB+ |
| VRAM | 4 GB (1.5B fp16) | 12 GB (7.2B INT8) |
| Disk | 20 GB | 50 GB+ |
| OS | Linux / macOS | Linux |

No CUDA toolkit required. PyTorch with CUDA runtime is sufficient.

---

## Project structure

```
flint/
├── core/
│   ├── model.py              # CompanionModel — state load/save, generation, LoRA apply
│   ├── session.py            # ConversationDB (SQLite), Session management
│   └── model_backends/       # rwkv_backend, onnx_backend, dummy_backend
├── lora/
│   ├── trainer.py            # LoRA training — pure PyTorch, no external framework
│   ├── pipeline.py           # Training pipeline — data selection, replay buffer
│   └── scheduler.py          # Nightly scheduler daemon
├── services/
│   ├── chat_service.py       # Session lifecycle
│   ├── reflection_service.py # Conversation summarisation and semantic indexing
│   ├── idle_reasoning.py     # Generates proactive questions during idle time
│   ├── dream_service.py      # Autonomous inner monologue generation
│   ├── backup_service.py     # State/adapter/DB backup and rotation
│   ├── model_registry.py     # Model discovery and metadata
│   └── scheduler_service.py  # Background job runner
├── web/
│   ├── server.py             # FastAPI app — REST + WebSocket
│   └── static/index.html     # Web UI (single file)
├── tools/
│   └── parser.py             # Conversation import (ChatGPT / Claude)
├── config.py                 # Default configuration (do not edit)
├── config.example.py         # Template for config_local.py
├── main.py                   # Terminal UI entry point
├── install.sh
├── start.sh
└── stop.sh
```

---

## License

GPLv3.0

## Beta candidate checklist

See `BETA_CANDIDATE_CHECKLIST.md` for the remaining items before a formal beta release.
