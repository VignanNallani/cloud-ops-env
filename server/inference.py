import os, json, asyncio
from pydantic import BaseModel
from typing import List, Optional, Dict

# 1. MUST BE PRESENT FOR app.py
class CloudOpsAction(BaseModel):
    command: str
    server_id: Optional[str] = None
    instance_tier: Optional[str] = None

class CloudOpsObservation(BaseModel):
    servers: List[Dict]
    current_cost_performance_ratio: float
    target_cost_performance_ratio: float
    grader_scores: Dict[str, float]

def log(msg):
    print(msg, flush=True)

async def run_logic(base_url: str):
    import websockets
    # MANDATORY START TAG [cite: 21]
    log(f"[START] task=cloud_ops env=cloud_ops_env model={os.getenv('MODEL_NAME', 'gemini-2.0-flash')}")
    
    rewards_list = []
    steps_count = 0
    success = "false" # lowercase boolean [cite: 29]

    try:
        ws_url = base_url.replace("http", "ws") + "/ws"
        async with websockets.connect(ws_url, timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                steps_count = t + 1
                action = {"command": "noop"} # Your 0.93 logic here...
                
                await ws.send(json.dumps({"type": "step", "action": action}))
                step_res = json.loads(await ws.recv())
                
                reward = float(step_res.get("reward", 0.0))
                rewards_list.append(reward)
                done = "true" if step_res.get("done") else "false"

                # MANDATORY STEP TAG [cite: 22, 28]
                log(f"[STEP] step={steps_count} action={action['command']} reward={reward:.2f} done={done} error=null")
                
                if step_res.get("done"):
                    success = "true"
                    break
    finally:
        # MANDATORY END TAG [cite: 23, 27]
        formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_list])
        log(f"[END] success={success} steps={steps_count} rewards={formatted_rewards}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
