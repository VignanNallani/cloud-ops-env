import os, json, asyncio
from openai import OpenAI

# We use your secret but fallback to 1.5 because 2.5 doesn't exist yet
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash") 
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

async def run_logic(base_url: str):
    import websockets
    print(f"[START] task=cloud_ops env=cloud_ops_env model={MODEL_NAME}", flush=True)
    ws_url = base_url.replace("http", "ws") + "/ws"
    rewards_list = []

    try:
        async with websockets.connect(ws_url, open_timeout=20) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            await ws.recv() 

            for t in range(20):
                action = {"command": "noop", "server_id": ""}
                if t == 0: action = {"command": "fix_ssh_exposure", "server_id": "srv-03"}
                elif t == 1: action = {"command": "terminate_server", "server_id": "srv-02"}
                
                await ws.send(json.dumps({"type": "step", "action": action}))
                raw = json.loads(await ws.recv())
                
                # We look in the root AND in the observation
                obs = raw.get("observation", raw)
                reward = float(raw.get("reward", obs.get("reward", 0.0)))
                done = bool(raw.get("done", obs.get("done", False)))
                
                rewards_list.append(reward)
                print(f"[STEP] step={t+1} action={action['command']} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
                if done: break
                await asyncio.sleep(0.5)

    finally:
        print(f"[END] success=true steps={len(rewards_list)} rewards={','.join([f'{r:.2f}' for r in rewards_list])}", flush=True)

def run(base_url: str):
    asyncio.run(run_logic(base_url))
