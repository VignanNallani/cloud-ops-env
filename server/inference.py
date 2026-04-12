import json, os, sys, asyncio
from openai import OpenAI
from typing import Any, Literal, Optional
from uuid import uuid4
from abc import ABC, abstractmethod
from enum import Enum

# Force hardware-level unbuffering
sys.stdout.reconfigure(line_buffering=True)

class SecurityStatus(str, Enum):
    """Posture for SSH / port 22 exposure."""
    SECURE = "secure"
    SSH_EXPOSED_WORLD = "ssh_exposed_world"  # Port 22 bound to 0.0.0.0/0

class Server:
    """One fleet member (virtual server)."""

    def __init__(
        self,
        id: str,
        cpu_utilization_percent: float,
        hourly_cost_usd: float,
        security_status: SecurityStatus,
        performance_units: float = 0.0,
        active: bool = True,
        ssh_listens_on: str = "127.0.0.1",
    ):
        self.id = id
        self.cpu_utilization_percent = cpu_utilization_percent
        self.hourly_cost_usd = hourly_cost_usd
        self.security_status = security_status
        self.performance_units = performance_units
        self.active = active
        self.ssh_listens_on = ssh_listens_on
        self._validate_ssh_consistency()

    def _validate_ssh_consistency(self):
        if self.security_status == SecurityStatus.SSH_EXPOSED_WORLD:
            if self.ssh_listens_on not in ("0.0.0.0", "0.0.0.0/0"):
                self.ssh_listens_on = "0.0.0.0"
        elif self.security_status == SecurityStatus.SECURE and self.ssh_listens_on in ("0.0.0.0", "0.0.0.0/0"):
            self.security_status = SecurityStatus.SSH_EXPOSED_WORLD

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "id": self.id,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "hourly_cost_usd": self.hourly_cost_usd,
            "security_status": self.security_status,
            "performance_units": self.performance_units,
            "active": self.active,
            "ssh_listens_on": self.ssh_listens_on,
        }

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

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "command": self.command,
            "server_id": self.server_id,
            "instance_tier": self.instance_tier,
        }

class CloudOpsObservation:
    """Observable fleet state and grading hints."""

    def __init__(
        self,
        summary_message: str = "",
        servers: Optional[list[Server]] = None,
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

SYSTEM_PROMPT = """You are a cloud operations and security auditor.

CRITICAL PRIORITIES (execute in this exact order):

Priority 1: If any server is ssh_exposed_world, you MUST use fix_ssh_exposure immediately. Security is the highest priority.

Priority 2: If a server is idle (CPU < 5%), you MUST use terminate_server. Eliminate waste.

Priority 3: Adjust the tier to match the target cost ratio. Optimize performance.

World vs private exposure in this simulation:
- `0.0.0.0` means the service is bound to all interfaces (world-reachable). This is insecure for SSH.

Available actions (pick exactly one per step):
- `noop`
- `terminate_server`: only valid for a server with `cpu_utilization_percent < 5.0` (idle)
- `fix_ssh_exposure`: for servers currently showing SSH exposure (`security_status == "ssh_exposed_world"`)
- `set_instance_tier`: set `instance_tier` to one of: "nano", "standard", "performance"

CRITICAL: Respond with RAW JSON only - NO markdown tags, NO ```json wrapper, NO extra text.
Output format: {"command": "...", "server_id": "...", "instance_tier": "standard"}
- Use empty string for `server_id` when using `noop`.
- `instance_tier` is only used with `set_instance_tier` (still include it in the JSON).
"""

def observation_to_prompt(obs: CloudOpsObservation) -> str:
    payload: dict[str, Any] = {
        "summary_message": obs.summary_message,
        "current_cost_performance_ratio": obs.current_cost_performance_ratio,
        "target_cost_performance_ratio": obs.target_cost_performance_ratio,
        "grader_scores": obs.grader_scores,
        "servers": [s.model_dump(mode="json") for s in obs.servers],
    }
    return json.dumps(payload, indent=2)

def _parse_action_or_error(text: str) -> CloudOpsAction:
    """Parse LLM response, falling back to noop if JSON is malformed."""
    try:
        data = json.loads(text)
        return CloudOpsAction(**data)
    except Exception as e:
        return CloudOpsAction(command="noop", server_id="", instance_tier="standard")

def _validate_action_or_error(obs: CloudOpsObservation, action: CloudOpsAction) -> str | None:
    """Return an error string if action is invalid for current observation."""
    if action.command == "noop":
        return None
    if action.command == "terminate_server":
        # Find the server and check if it's idle
        for server in obs.servers:
            if server.id == action.server_id:
                if server.cpu_utilization_percent >= 5.0:
                    return f"Server {action.server_id} is not idle (CPU {server.cpu_utilization_percent}% >= 5%)"
                if not server.active:
                    return f"Server {action.server_id} is already terminated"
                return None
        return f"Server {action.server_id} not found"
    if action.command == "fix_ssh_exposure":
        # Find the server and check if it has SSH exposure
        for server in obs.servers:
            if server.id == action.server_id:
                if server.security_status != SecurityStatus.SSH_EXPOSED_WORLD:
                    return f"Server {action.server_id} does not have SSH exposure"
                return None
        return f"Server {action.server_id} not found"
    if action.command == "set_instance_tier":
        # Find the server and check if it's active
        for server in obs.servers:
            if server.id == action.server_id:
                if not server.active:
                    return f"Server {action.server_id} is terminated"
                return None
        return f"Server {action.server_id} not found"
    return f"Error: Unsupported command {action.command!r}"

def select_action(
    obs: CloudOpsObservation,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> CloudOpsAction:
    """Call OpenAI and robustly return a validated action (self-correction)."""
    # Use HF_TOKEN as primary API key source
    api_key = os.getenv('HF_TOKEN') or os.getenv('OPENAI_API_KEY')
    api = client or OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")

    base_user_content = observation_to_prompt(obs)
    last_error: str | None = None

    for _ in range(max_retries):
        user_content = base_user_content

        if last_error:
            user_content = (
                f"{base_user_content}\n\nValidation feedback from the environment:\n{last_error}\n"
                "Return corrected JSON only."
            )

        resp = api.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        action = _parse_action_or_error(resp.choices[0].message.content)
        last_error = _validate_action_or_error(obs, action)
        if last_error is None:
            return action

    # If we get here, keep going with a safe noop rather than crashing.
    return CloudOpsAction(command="noop", server_id="", instance_tier="standard")

async def run_logic(base_url: str):
    import aiohttp, websockets
    
    # 1. THE START TAG - First line of output
    print(f'[START] task=cloud_ops', flush=True)
    
    rewards = []
    steps = 0
    success = False
    
    try:
        ws_url = base_url.replace("http", "ws") + "/ws"
        async with websockets.connect(ws_url) as ws:
            # Reset environment
            await ws.send(json.dumps({"type": "reset", "seed": 0}))
            res = json.loads(await ws.recv())
            obs_data = res.get("observation", {})
            
            # Agent Loop (Max 20 steps)
            for t in range(20):
                # Process observation
                obs = CloudOpsObservation(**obs_data)
                obs.servers = [Server(**s) for s in obs_data.get("servers", [])]
                
                # Get and execute action
                action = select_action(obs)
                await ws.send(json.dumps({"type": "step", "action": action.model_dump()}))
                
                # Parse response
                step_res = json.loads(await ws.recv())
                obs_data = step_res.get("observation", {})
                reward = step_res.get("reward", 0.0)
                done = step_res.get("done", False)
                
                rewards.append(reward)
                steps = t + 1
                success = done
                
                # 2. THE STEP TAG - Clean and precise
                print(f'[STEP] step={steps} reward={reward:.2f} done={str(done).lower()}', flush=True)
                if done: break
    except Exception:
        pass
    finally:
        # 3. THE END TAG - Exactly one, guaranteed by 'finally'
        score = sum(rewards)
        print(f'[END] task=cloud_ops score={score:.2f} steps={steps}', flush=True)

def run(base_url: str):
    # Bridge between FastAPI and async logic
    asyncio.run(run_logic(base_url))
