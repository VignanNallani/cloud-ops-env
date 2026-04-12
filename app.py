from fastapi import FastAPI
import uvicorn, asyncio, env
from inference import run, CloudOpsAction, CloudOpsObservation
from openenv.core.env_server.http_server import create_app

# Initialize OpenEnv wrapper
app = create_app(
    env.CloudOpsEnvironment,
    CloudOpsAction,
    CloudOpsObservation,
    env_name="cloud_ops_env"
)

@app.post('/reset')
async def trigger_run(request: dict = None):
    # This runs inference in background so HTTP request doesn't timeout
    asyncio.create_task(asyncio.to_thread(run, 'http://localhost:8000'))
    return {"status": "VIGNAN_PHASE2_ACTIVE"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
