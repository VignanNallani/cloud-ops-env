import os, json, asyncio
from pydantic import BaseModel
from typing import List, Optional, Dict
from openai import OpenAI

# 1. MANDATORY MODELS FOR app.py
class CloudOpsAction(BaseModel):
    command: str
    server_id: Optional[str] = None
    instance_tier: Optional[str] = None

class CloudOpsObservation(BaseModel):
    servers: List[Dict]
    current_cost_performance_ratio: float
    target_cost_performance_ratio: float
    grader_scores: Dict[str, float]

# 2. REQUIRED ENVIRONMENT VARIABLES [cite: 9-18]
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

def log(msg):
    print(msg, flush=True)

# Ensure this check is at the top of your script
if not HF_TOKEN:
    # This prevents the 400 error by stopping the script before the call
    log("CRITICAL ERROR: HF_TOKEN environment variable is missing in Space Settings!")
    raise ValueError("HF_TOKEN is mandatory")

# 3. INITIALIZE OPENAI CLIENT [cite: 47-51]
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

async def run_logic(base_url: str):
    import websockets
    # [START] TAG [cite: 21, 25]
    log(f"[START] task=cloud_ops env=cloud_ops_env model={MODEL_NAME}")
    
    ws_url = base_url.replace("http", "ws") + "/ws"
    rewards_list = []
    steps_count = 0
    success = "false"

    try:
        async with websockets.connect(ws_url, open_timeout=15) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                steps_count = t + 1
                servers = obs.get("servers", [])
                
                # --- MANDATORY LLM DECISION MAKING [cite: 6, 52-60] ---
                # We ask the LLM to analyze the state and provide the next best action
                prompt = f"System State: {json.dumps(obs)}. Provide the next command (fix_ssh_exposure, terminate_server, or set_instance_tier) to optimize cost and security. Return ONLY a JSON object: {{'command': '...', 'server_id': '...'}}"
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                action_dict = json.loads(response.choices[0].message.content)

                # --- EXECUTE STEP ---
                await ws.send(json.dumps({"type": "step", "action": action_dict}))
                step_res = json.loads(await ws.recv())
                
                # --- UPDATE STATE & LOGGING ---
                obs = step_res.get("observation", {})
                reward = float(step_res.get("reward", 0.0))
                rewards_list.append(reward)
                done_bool = "true" if step_res.get("done") else "false" [cite: 29]
                err = step_res.get("last_action_error", "null") [cite: 30]
                if err is None: err = "null"

                # [STEP] TAG [cite: 22, 28]
                log(f"[STEP] step={steps_count} action={action_dict.get('command', 'noop')} reward={reward:.2f} done={done_bool} error={err}")
                
                if step_res.get("done"):
                    success = "true"
                    break

    except Exception as e:
        log(f"Pipeline Error: {str(e)}")
    finally:
        # [END] TAG [cite: 23, 27, 28]
        formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_list])
        log(f"[END] success={success} steps={steps_count} rewards={formatted_rewards}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
