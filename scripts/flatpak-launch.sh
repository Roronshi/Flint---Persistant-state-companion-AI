#!/usr/bin/env bash
cd /app/flint
exec python3 -m uvicorn web.server:app --host 0.0.0.0 --port 8000
