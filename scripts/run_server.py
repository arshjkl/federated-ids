#!/usr/bin/env python
"""
Run script for Federated Learning IDS Dashboard Server.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env if exists (without python-dotenv dependency)
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())
    print(f"Loaded environment from {env_file}")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting server on {host}:{port} (debug={debug})")
    uvicorn.run(
        "src.dashboard.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )