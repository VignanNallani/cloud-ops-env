from fastapi import FastAPI
import uvicorn
import asyncio
import os, sys
import env
from inference import run, CloudOpsAction, CloudOpsObservation
from openenv.core.env_server.http_server import create_app

# 1. Host the actual environment endpoints (WebSocket, /step, etc.)
app = create_app(
    env.CloudOpsEnvironment,
    CloudOpsAction,
    CloudOpsObservation,
    env_name="cloud_ops_env",
    max_concurrent_envs=1,
)

# 2. Add the trigger endpoint with the mandatory [START] tag
@app.post('/reset')
async def reset_endpoint(request: dict = None):
    # This MUST be the first line of output the validator sees
    sys.stdout.write("[START] task=cloud_ops\n")
    sys.stdout.flush()
    
    try:
        # Run the agent (it connects to the WebSocket hosted above)
        await asyncio.to_thread(run, base_url='http://0.0.0.0:8000')
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # log_level="error" is what silences the Uvicorn INFO noise
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
