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

# Change '/reset' to '/trigger' to avoid the library conflict
@app.post('/trigger')
async def trigger_endpoint(request: dict = None):
    try:
        # Launch inference in background to prevent timeout
        asyncio.create_task(asyncio.to_thread(run, base_url='http://localhost:8000'))
        return {"status": "VIGNAN_ELITE_V34_ACTIVE"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
