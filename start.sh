#!/usr/bin/env bash
set -euo pipefail
cd /home/userul/.hermes/apps/boneplanner-cad
exec /home/userul/.hermes/apps/boneplanner-cad/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8121
