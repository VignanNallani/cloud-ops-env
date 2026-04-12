import json, os, sys, asyncio
from openai import OpenAI
from typing import Any, Literal, Optional
from enum import Enum

class CloudOpsAction:
    """Agent command for fleet operations."""
    def __init__(
        self,
        command: Literal["noop", "terminate_server", "fix_ssh_exposure", "set_instance_tier"] = "noop",
        server_id: str = "",
        instance_tier: Literal["nano", "standard", "performance"] = "standard",
    ):
        self.command = command
        self.server_id = server_id
        self.instance_tier = instance_tier

class CloudOpsObservation:
    """Observable fleet state and grading hints."""
    def __init__(
        self,
        summary_message: str = "",
        servers: Optional[list] = None,
        current_cost_performance_ratio: float = 0.0,
        target_cost_performance_ratio: float = 0.0,
        grader_scores: Optional[dict[str, float]] = None,
        done: bool = False,
        reward: float = 0.0,
    ):
        self.summary_message = summary_message
        self.servers = servers or []
        self.current_cost_performance_ratio = current_cost_performance_ratio
        self.target_cost_performance_ratio = target_cost_performance_ratio
        self.grader_scores = grader_scores or {}
        self.done = done
        self.reward = reward

def log(msg):
    print(msg, flush=True)

async def run_logic(base_url: str):
    import websockets
    # 1. GRADER REQUIREMENT: START TAG
    log("[START] task=cloud_ops")
    
    ws_url = base_url.replace("http", "ws") + "/ws"
    api = OpenAI(api_key=os.getenv('HF_TOKEN'), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                # --- THE 0.93 LOGIC ---
                # 1. Check Security
                action = {"command": "noop"}
                for s in obs.get("servers", []):
                    if s.get("security_status") == "ssh_exposed_world":
                        action = {"command": "fix_ssh_exposure", "server_id": s['id']}
                        break
                    if s.get("cpu_utilization_percent", 100) < 5.0:
                        action = {"command": "terminate_server", "server_id": s['id']}
                        break
                
                # 2. Optimization (Cost/Performance Ratio)
                if action["command"] == "noop":
                    ratio = obs.get("current_cost_performance_ratio", 0)
                    target = obs.get("target_cost_performance_ratio", 0)
                    if ratio > target * 1.1:
                        action = {"command": "set_instance_tier", "server_id": obs['servers'][0]['id'], "instance_tier": "nano"}
                
                # --- EXECUTE STEP ---
                await ws.send(json.dumps({"type": "step", "action": action}))
                step_res = json.loads(await ws.recv())
                obs = step_res.get("observation", {})
                reward = step_res.get("reward", 0.0)
                
                # 2. GRADER REQUIREMENT: STEP TAG
                log(f"[STEP] step={t+1} reward={reward:.2f}")
                
                if step_res.get("done"): break

            # 3. GRADER REQUIREMENT: END TAG
            final_score = obs.get("grader_scores", {}).get("hard", 0.0)
            log(f"[END] task=cloud_ops score={final_score:.2f}")
            
    except Exception as e:
        log(f"[END] task=cloud_ops error={str(e)}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
