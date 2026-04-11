# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Cloud Ops Env Environment.

This module creates an HTTP server that exposes the CloudOpsEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python app.py

Direct execution calls ``main()`` with optional ``--port``.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    pip install openenv\n'"
    ) from e

# Import FastAPI for decorators
from fastapi import FastAPI
import sys
import asyncio
import logging

# Hijack the root logger to force output visibility
logging.basicConfig(level=logging.DEBUG, force=True)

# Confirm PYTHONUNBUFFERED=1 is set
os.environ.setdefault('PYTHONUNBUFFERED', '1')

# Import classes from parent directory
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import CloudOpsAction, CloudOpsObservation, run
import env

# Create the app with web interface and README integration
app = create_app(
    env.CloudOpsEnvironment,
    CloudOpsAction,
    CloudOpsObservation,
    env_name="cloud_ops_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Add custom reset endpoint to trigger inference logic
@app.post('/reset')
async def reset_endpoint():
    """Reset endpoint that triggers inference logic."""
    try:
        # CRITICAL: Call run function to trigger inference logic
        # Use asyncio to avoid blocking FastAPI event loop
        await asyncio.to_thread(run, base_url='http://0.0.0.0:8000')
        
        # Gold Standard: Ensure all stdout/stderr is flushed immediately
        sys.stdout.flush()
        sys.stderr.flush()
        
        return {"status": "success", "message": "Inference triggered and flushed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . app
        uv run --project . app --port 8001
        python app.py

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen to (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn app:app --workers 4
    """
    import uvicorn

    # Gold Standard: Disable all uvicorn and FastAPI default logging
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
