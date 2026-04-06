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
from typing import Any

from openai import OpenAI

from .env import CloudOpsAction, CloudOpsObservation, SecurityStatus


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
    api_base_url = os.environ.get("API_BASE_URL")
    api = client or OpenAI(base_url=api_base_url) if api_base_url else OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

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

    from .client import CloudOpsEnv

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
