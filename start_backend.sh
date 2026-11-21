#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
export PYTHONPATH="$DIR:$PYTHONPATH"
exec "$PYTHON" -m uvicorn src.dashboard.main:app --reload --host 0.0.0.0 --port "${PORT:-18003}"
