# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM policy using the OpenAI client: maps observations to ``CloudOpsAction``.

Requires ``OPENAI_API_KEY``. Model defaults to ``gemini-2.5-flash`` (override with ``OPENAI_MODEL``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional
from uuid import uuid4
from abc import ABC, abstractmethod
from enum import Enum

from openai import OpenAI


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


def _parse_action_or_error(text: str) -> CloudOpsAction:
    """Parse LLM response, falling back to noop if JSON is malformed."""
    try:
        data = json.loads(text)
        return CloudOpsAction(**data)
    except Exception as e:
        print(f"Error parsing action: {e}")
        return CloudOpsAction(command="noop", server_id="", instance_tier="standard")


def _validate_action_or_error(obs: CloudOpsObservation, action: CloudOpsAction) -> str | None:
    """Return an error string if the action is invalid for the current observation."""
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
        action = _parse_action_or_error(resp.choices[0].message.content)
        last_error = _validate_action_or_error(obs, action)
        if last_error is None:
            return action

    # If we get here, keep going with a safe noop rather than crashing.
    return CloudOpsAction(command="noop", server_id="", instance_tier="standard")


class CloudOpsEnv:
    """Client for the Cloud Ops & Security Auditor environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._session = None

    async def __aenter__(self):
        import aiohttp
        self._session = aiohttp.ClientSession()
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def connect(self):
        """Establish WebSocket connection."""
        import asyncio
        import websockets
        
        self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        try:
            self._ws = await asyncio.wait_for(websockets.connect(self._ws_url), timeout=10)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e

    async def reset(self, seed: int = 0):
        """Reset environment and return initial observation."""
        reset_msg = {"type": "reset", "seed": seed}
        await self._ws.send(json.dumps(reset_msg))
        response = await self._ws.recv()
        data = json.loads(response)
        
        obs_data = data.get("observation", {})
        observation = CloudOpsObservation(
            summary_message=obs_data.get("summary_message", ""),
            servers=[Server(**s) for s in obs_data.get("servers", [])],
            current_cost_performance_ratio=obs_data.get("current_cost_performance_ratio", 0.0),
            target_cost_performance_ratio=obs_data.get("target_cost_performance_ratio", 0.0),
            grader_scores=obs_data.get("grader_scores", {}),
            done=data.get("done", False),
            reward=data.get("reward", 0.0),
        )
        
        from types import SimpleNamespace
        return SimpleNamespace(observation=observation)

    async def step(self, action: CloudOpsAction):
        """Execute action and return result."""
        step_msg = {
            "type": "step",
            "action": action.model_dump(mode="json")
        }
        await self._ws.send(json.dumps(step_msg))
        response = await self._ws.recv()
        data = json.loads(response)
        
        obs_data = data.get("observation", {})
        observation = CloudOpsObservation(
            summary_message=obs_data.get("summary_message", ""),
            servers=[Server(**s) for s in obs_data.get("servers", [])],
            current_cost_performance_ratio=obs_data.get("current_cost_performance_ratio", 0.0),
            target_cost_performance_ratio=obs_data.get("target_cost_performance_ratio", 0.0),
            grader_scores=obs_data.get("grader_scores", {}),
            done=data.get("done", False),
            reward=data.get("reward", 0.0),
        )
        
        from types import SimpleNamespace
        return SimpleNamespace(observation=observation, reward=data.get("reward", 0.0), done=data.get("done", False))


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
