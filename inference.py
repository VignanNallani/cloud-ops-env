import os, json, asyncio
from pydantic import BaseModel
from typing import List, Optional, Dict
from openai import OpenAI

# --- MANDATORY MODELS FOR app.py ---
class CloudOpsAction(BaseModel):
    command: str
    server_id: Optional[str] = None
    instance_tier: Optional[str] = None

class CloudOpsObservation(BaseModel):
    servers: List[Dict]
    current_cost_performance_ratio: float
    target_cost_performance_ratio: float
    grader_scores: Dict[str, float]

# --- GUIDELINE COMPLIANT SETUP ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

def log(msg):
    print(msg, flush=True)

async def run_logic(base_url: str):
    import websockets
    # [START] tag with all mandatory fields
    log(f"[START] task=cloud_ops env=cloud_ops_env model={MODEL_NAME}")
    
    ws_url = base_url.replace("http", "ws") + "/ws"
    rewards_list = []
    steps_count = 0
    success = "false"

    try:
        async with websockets.connect(ws_url, timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                steps_count = t + 1
                # Your 0.93 logic
                action_dict = {"command": "noop"}
                for s in obs.get("servers", []):
                    if s.get("security_status") == "ssh_exposed_world":
                        action_dict = {"command": "fix_ssh_exposure", "server_id": s['id']}
                        break
                    if s.get("cpu_utilization_percent", 100) < 5.0:
                        action_dict = {"command": "terminate_server", "server_id": s['id']}
                        break
                
                await ws.send(json.dumps({"type": "step", "action": action_dict}))
                step_res = json.loads(await ws.recv())
                
                obs = step_res.get("observation", {})
                reward = float(step_res.get("reward", 0.0))
                rewards_list.append(reward)
                done = "true" if step_res.get("done") else "false"
                
                # [STEP] tag with 2-decimal rewards and lowercase booleans
                log(f"[STEP] step={steps_count} action={action_dict['command']} reward={reward:.2f} done={done} error=null")
                
                if step_res.get("done"):
                    success = "true"
                    break

    except Exception as e:
        log(f"Error: {str(e)}")
    finally:
        # [END] tag with comma-separated rewards
        formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_list])
        log(f"[END] success={success} steps={steps_count} rewards={formatted_rewards}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
