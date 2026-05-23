#!/usr/bin/env bash
set -euo pipefail
cd /home/userul/.hermes/apps/3dmedicalplanner
exec /home/userul/.hermes/apps/3dmedicalplanner/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8121
