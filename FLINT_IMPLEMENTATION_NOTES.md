# Flint implementation notes

This pass establishes the first working Flint backbone inside the codebase.

## Implemented in this pass

- Added `core/app_state.py` to centralize runtime state.
- Introduced service-layer modules:
  - `services/model_registry.py`
  - `services/chat_service.py`
  - `services/reflection_service.py`
  - `services/scheduler_service.py`
- Refactored `web/server.py` to use app state and services.
- Expanded `core/session.py` from a simple chat log into the first Flint data model with:
  - companions
  - models
  - initiative profiles
  - conversation blocks
  - summaries
  - semantic memory
  - reflections
  - outreach candidates/events
  - background job runs
- Kept RWKV-first and one-active-model semantics.
- Added a first background-job scheduler for ingest/summarize/synthesis/memory refresh.
- Added reflection generation and gating pipeline.
- Added `/api/outreach` and `/api/reflect/run` endpoints.
- Added WebSocket stop handling.

## Important limitations of this pass

- The reflection engine is currently extractive / heuristic. It does **not yet** run true idle reasoning through the RWKV model.
- Model switching is still registry-aware rather than full load/unload runtime management.
- Frontend UX has not yet been rebuilt to expose all new Flint concepts cleanly.
- LoRA remains scheduled separately and is not yet fully integrated with adapter lineage tables.
- No formal automated test suite has been added yet.

## Why this is still a meaningful step

This pass creates the internal architecture needed for the later product experience:
- background cognition can now exist as first-class system behavior
- reflections are stored as real intermediate artifacts
- outreach no longer has to be generated directly from raw history
- initiative profiles can influence system behavior later without a schema rewrite
- the codebase is now much better positioned for Linux-first packaging and future desktop UX work

## Recommended next implementation step

1. Add repository-style wrappers or service methods for adapter lineage and runtime snapshots.
2. Rebuild the frontend around Flint status, initiative mode and outreach visibility.
3. Add migrations / tests.
4. Add true model-mediated idle reasoning for reflections.
