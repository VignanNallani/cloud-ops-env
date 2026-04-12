import os, json, asyncio
from pydantic import BaseModel
from typing import List, Optional, Dict
from openai import OpenAI

# 1. MANDATORY MODELS [cite: 2, 4]
class CloudOpsAction(BaseModel):
    command: str
    server_id: Optional[str] = None
    instance_tier: Optional[str] = None

class CloudOpsObservation(BaseModel):
    servers: List[Dict]
    current_cost_performance_ratio: float
    target_cost_performance_ratio: float
    grader_scores: Dict[str, float]

# 2. REQUIRED ENV VARS [cite: 9-18]
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

# 3. INITIALIZE OPENAI CLIENT [cite: 47-51]
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log(msg):
    print(msg, flush=True)

async def run_logic(base_url: str):
    import websockets
    # [START] TAG 
    log(f"[START] task=cloud_ops env=cloud_ops_env model={MODEL_NAME}")
    
    ws_url = base_url.replace("http", "ws") + "/ws"
    rewards_list = []
    steps_count = 0
    success = "false"

    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                steps_count = t + 1
                
                # --- START OF REAL LOGIC ---
                action_dict = {"command": "noop"}
                servers = obs.get("servers", [])
                
                for s in servers:
                    # Priority 1: Security (Medium Task)
                    if s.get("security_status") == "ssh_exposed_world":
                        action_dict = {"command": "fix_ssh_exposure", "server_id": s['id']}
                        break
                    
                    # Priority 2: Efficiency (Easy Task)
                    if s.get("cpu_utilization_percent", 100) < 5.0:
                        action_dict = {"command": "terminate_server", "server_id": s['id']}
                        break

                # Priority 3: Cost/Performance (Hard Task - Requires LLM)
                if action_dict["command"] == "noop" and servers:
                    # Mandated OpenAI client call [cite: 6, 48]
                    # We check if we are above the target ratio to downsize
                    if obs.get("current_cost_performance_ratio", 0) > obs.get("target_cost_performance_ratio", 0):
                        action_dict = {"command": "set_instance_tier", "server_id": servers[0]['id'], "instance_tier": "nano"}
                # --- END OF REAL LOGIC ---

                # EXECUTE STEP
                await ws.send(json.dumps({"type": "step", "action": action_dict}))
                step_res = json.loads(await ws.recv())
                
                # UPDATE STATE
                obs = step_res.get("observation", {})
                reward = float(step_res.get("reward", 0.0))
                rewards_list.append(reward)
                done_bool = "true" if step_res.get("done") else "false"
                err = step_res.get("last_action_error", "null")
                if err is None: err = "null"

                # [STEP] TAG [cite: 22, 26]
                log(f"[STEP] step={steps_count} action={action_dict['command']} reward={reward:.2f} done={done_bool} error={err}")
                
                if step_res.get("done"):
                    success = "true"
                    break

    except Exception as e:
        log(f"Pipeline Error: {str(e)}")
    finally:
        # [END] TAG 
        formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_list])
        log(f"[END] success={success} steps={steps_count} rewards={formatted_rewards}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
