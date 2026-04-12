import os, json, sys, asyncio
from openai import OpenAI

# 1. REQUIRED ENVIRONMENT VARIABLES WITH DEFAULTS 
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required ")

# 2. INITIALIZE OPENAI CLIENT [cite: 8]
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log(msg):
    print(msg, flush=True)

async def run_logic(base_url: str):
    import websockets
    # 3. MANDATORY START TAG 
    # Fields: task, env, model
    log(f"[START] task=cloud_ops env=cloud_ops_env model={MODEL_NAME}")
    
    ws_url = base_url.replace("http", "ws") + "/ws"
    rewards_list = []
    steps_count = 0
    success = "false" # lowercase boolean 

    try:
        async with websockets.connect(ws_url, timeout=10) as ws:
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs = res.get("observation", {})

            for t in range(20):
                steps_count = t + 1
                # Your existing 0.93 optimization logic here...
                action_dict = {"command": "noop"} 
                
                # EXECUTE STEP
                await ws.send(json.dumps({"type": "step", "action": action_dict}))
                step_res = json.loads(await ws.recv())
                
                # DATA EXTRACTION
                reward = float(step_res.get("reward", 0.0))
                rewards_list.append(reward)
                done = "true" if step_res.get("done") else "false"
                error_msg = step_res.get("last_action_error", "null") # raw string or null [cite: 6]
                if error_msg is None: error_msg = "null"

                # 4. MANDATORY STEP TAG (Immediately after env.step) [cite: 2, 3]
                # reward must be 2 decimal places, done is lowercase 
                log(f"[STEP] step={steps_count} action={action_dict['command']} reward={reward:.2f} done={done} error={error_msg}")
                
                if step_res.get("done"): 
                    success = "true"
                    break

    except Exception as e:
        log(f"Internal Error: {str(e)}")
    finally:
        # 5. MANDATORY END TAG (Always emitted, even on exception) [cite: 2, 4]
        # rewards is a comma-separated list of 2-decimal floats [cite: 2, 5]
        formatted_rewards = ",".join([f"{r:.2f}" for r in rewards_list])
        log(f"[END] success={success} steps={steps_count} rewards={formatted_rewards}")

def run(base_url: str):
    asyncio.run(run_logic(base_url))
