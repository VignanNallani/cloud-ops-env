# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM policy using the OpenAI client: maps observations to ``CloudOpsAction``.

Requires ``OPENAI_API_KEY``. Model defaults to ``meta-llama/Llama-3.2-3B-Instruct`` (override with ``OPENAI_MODEL``).
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from typing import Any, Dict, Literal

from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# --- Copied Classes from env.py ---

class SecurityStatus(str, Enum):
    """Posture for SSH / port 22 exposure."""

    SECURE = "secure"
    SSH_EXPOSED_WORLD = "ssh_exposed_world"  # Port 22 bound to 0.0.0.0/0


class Server(BaseModel):
    """One fleet member (virtual server)."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    id: str = Field(..., description="Stable server identifier")
    cpu_utilization_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Current CPU utilization (%)"
    )
    hourly_cost_usd: float = Field(..., ge=0.0, description="Hourly cost (USD)")
    security_status: SecurityStatus = Field(..., description="Security posture")
    performance_units: float = Field(
        default=0.0,
        ge=0.0,
        description="Abstract capacity used for cost-vs-performance ratio",
    )
    active: bool = Field(default=True, description="False if instance terminated")
    ssh_listens_on: str = Field(
        default="127.0.0.1",
        description="Bind address for SSH (0.0.0.0 = world-reachable)",
    )

    @model_validator(mode="after")
    def _ssh_consistent(self) -> Server:
        if self.security_status == SecurityStatus.SSH_EXPOSED_WORLD:
            if self.ssh_listens_on not in ("0.0.0.0", "0.0.0.0/0"):
                self.ssh_listens_on = "0.0.0.0"
        elif self.security_status == SecurityStatus.SECURE and self.ssh_listens_on in (
            "0.0.0.0",
            "0.0.0.0/0",
        ):
            self.security_status = SecurityStatus.SSH_EXPOSED_WORLD
        return self


TIER_SPECS: dict[str, tuple[float, float]] = {
    "nano": (0.02, 12.0),
    "standard": (0.06, 42.0),
    "performance": (0.14, 110.0),
}


class CloudOpsAction(Action):
    """Agent command for fleet operations."""

    command: Literal[
        "noop",
        "terminate_server",
        "fix_ssh_exposure",
        "set_instance_tier",
    ] = Field("noop", description="High-level operation")
    server_id: str = Field(
        default="",
        description="Target server id for terminate / fix_ssh / set_instance_tier",
    )
    instance_tier: Literal["nano", "standard", "performance"] = Field(
        "standard",
        description="Tier for set_instance_tier (updates cost & performance units)",
    )


class CloudOpsObservation(Observation):
    """Observable fleet state and grading hints."""

    summary_message: str = Field(
        default="", description="Short natural-language status for the agent"
    )
    servers: list[Server] = Field(default_factory=list, description="Current fleet")
    current_cost_performance_ratio: float = Field(
        default=0.0,
        description="sum(hourly_cost) / sum(performance_units) over active servers",
    )
    target_cost_performance_ratio: float = Field(
        default=0.0,
        description="Hard objective: match this ratio within tolerance",
    )
    grader_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-task achievement in [0, 1] (easy, medium, hard)",
    )


# --- Copied CloudOpsEnv class from client.py ---

class CloudOpsEnv(EnvClient[CloudOpsAction, CloudOpsObservation, State]):
    """
    Client for the Cloud Ops & Security Auditor environment.

    Example:
        >>> async with CloudOpsEnv(base_url="http://localhost:8000") as client:
        ...     r = await client.reset(seed=0)
        ...     assert r.observation.servers
    """

    def _step_payload(self, action: CloudOpsAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CloudOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = CloudOpsObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# --- Original inference.py code ---

SYSTEM_PROMPT = """You are a cloud operations and security auditor.

Goal:
1. Reduce cost/performance ratio by matching `target_cost_performance_ratio` using `set_instance_tier`.
2. Fix security by eliminating world-reachable SSH exposure (port 22 bound to 0.0.0.0).
3. Make the episode end by also completing the easy objective: terminate idle servers (CPU < 5%).

World vs private exposure in this simulation:
- `0.0.0.0` means the service is bound to all interfaces (world-reachable). This is insecure for SSH.

Available actions (pick exactly one per step):
- `noop`
- `terminate_server`: only valid for a server with `cpu_utilization_percent < 5.0` (idle)
- `fix_ssh_exposure`: for servers currently showing SSH exposure (`security_status == "ssh_exposed_world"`)
- `set_instance_tier`: set `instance_tier` to one of: "nano", "standard", "performance"

Respond with a single JSON object only (no markdown, no extra text):
{"command": "...", "server_id": "...", "instance_tier": "standard"}
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


def parse_action_json(text: str) -> CloudOpsAction:
    """Parse a model response into a CloudOpsAction.

    Accepts raw JSON or a JSON object inside a code block.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Be forgiving: try to extract the first JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    data = json.loads(text)
    return CloudOpsAction(
        command=data.get("command", "noop"),
        server_id=data.get("server_id", "") or "",
        instance_tier=data.get("instance_tier", "standard") or "standard",
    )

def _validate_action_or_error(obs: CloudOpsObservation, action: CloudOpsAction) -> str | None:
    """Return an error message if action is invalid; otherwise None."""
    valid_ids = {s.id for s in obs.servers}

    if action.command == "noop":
        if action.server_id:
            # Tolerate but normalize message: env will treat server_id as ignored.
            return "Validation error: For `noop`, `server_id` must be an empty string."
        return None

    if not action.server_id or action.server_id not in valid_ids:
        return (
            f"Error: Server ID {action.server_id!r} does not exist. "
            "Please try again based on the provided list."
        )

    s_by_id = {s.id: s for s in obs.servers}
    s = s_by_id[action.server_id]

    if not s.active:
        return f"Error: Server ID {action.server_id!r} is already terminated (inactive)."

    if action.command == "terminate_server":
        if s.cpu_utilization_percent >= 5.0:
            return (
                f"Error: terminate_server is only allowed for idle servers (CPU < 5.0%). "
                f"Server {s.id!r} has CPU={s.cpu_utilization_percent:.1f}."
            )
        return None

    if action.command == "fix_ssh_exposure":
        # If already secure, it's not harmful, but it won't progress medium. Still valid.
        return None

    if action.command == "set_instance_tier":
        if action.instance_tier not in ("nano", "standard", "performance"):
            return f"Error: instance_tier {action.instance_tier!r} is invalid."
        return None

    return f"Error: Unsupported command {action.command!r}."


def select_action(
    obs: CloudOpsObservation,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> CloudOpsAction:
    """Call OpenAI and robustly return a validated action (self-correction)."""
    # Force correct Gemini endpoint and key
    api_key = os.environ.get("OPENAI_API_KEY")
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

        raw = resp.choices[0].message.content or "{}"
        try:
            action = parse_action_json(raw)
        except Exception as e:  # noqa: BLE001 - need to keep message for the retry loop
            last_error = f"Error: Could not parse action JSON ({type(e).__name__}). Please output valid JSON."
            continue

        last_error = _validate_action_or_error(obs, action)
        if last_error is None:
            return action

    # If we get here, keep going with a safe noop rather than crashing.
    return CloudOpsAction(command="noop", server_id="", instance_tier="standard")


def run_episode_demo(base_url: str, seed: int = 0, max_steps: int = 20) -> None:
    """Connect to a running OpenEnv server and roll out one greedy LLM episode (async loop)."""
    import asyncio

    async def _run() -> None:
        async with CloudOpsEnv(base_url=base_url) as env:
            result = await env.reset(seed=seed)
            episode_id = f"ep_{seed}_{hash(base_url) % 10000}"
            print(f"[START] Episode {episode_id}")
            print(f"[INIT] {result.observation.summary_message[:120]}")
            
            total_reward = 0
            for t in range(max_steps):
                action = select_action(result.observation)
                print(f"[STEP {t + 1}] Action: {action.command} | Server: {action.server_id} | Tier: {action.instance_tier}")
                result = await env.step(action)
                total_reward += result.reward
                print(f"[STEP {t + 1}] Reward: {result.reward:.3f} | Cumulative: {total_reward:.3f} | Done: {result.done}")
                if result.done:
                    break
            
            print(f"[END] Final Score: {total_reward:.3f} | Steps: {t + 1}")

    asyncio.run(_run())


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    run_episode_demo(url)
