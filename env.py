# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cloud Ops & Security Auditor - environment core.

Fleet management simulation with three graded objectives (easy / medium / hard).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


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


# --- Graders -----------------------------------------------------------------


class TaskGrader(ABC):
    """Maps fleet state to [0, 1] achievement with partial credit."""

    name: str

    @abstractmethod
    def score(
        self,
        servers: list[Server],
        ctx: dict[str, Any],
    ) -> float:
        pass

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))


class EasyIdleTerminationGrader(TaskGrader):
    """
    Easy: terminate at least one server that was idle (CPU < 5%) at episode start.
    Partial credit: fraction of initially-idle active servers that have been terminated.
    """

    name = "easy"

    def score(self, servers: list[Server], ctx: dict[str, Any]) -> float:
        idle_ids: set[str] = ctx.get("initial_idle_server_ids", set())
        if not idle_ids:
            return 1.0
        terminated_idle = sum(
            1 for s in servers if (not s.active) and s.id in idle_ids
        )
        return self._clamp01(terminated_idle / max(1, len(idle_ids)))


class MediumSSHGrader(TaskGrader):
    """
    Medium: no world-exposed SSH (port 22 to 0.0.0.0).
    Partial credit: fraction of servers that were vulnerable at start and are now secure.
    """

    name = "medium"

    def score(self, servers: list[Server], ctx: dict[str, Any]) -> float:
        was_vuln: set[str] = ctx.get("initial_vulnerable_server_ids", set())
        if not was_vuln:
            return 1.0

        def is_fixed(s: Server) -> bool:
            if not s.active:
                return False
            ok_bind = s.ssh_listens_on not in ("0.0.0.0", "0.0.0.0/0")
            return s.security_status == SecurityStatus.SECURE and ok_bind

        fixed = sum(1 for s in servers if s.id in was_vuln and is_fixed(s))
        return self._clamp01(fixed / max(1, len(was_vuln)))


class HardCostPerformanceGrader(TaskGrader):
    """
    Hard: rebalance fleet so aggregate cost/performance ratio is near target.
    Partial credit: 1 - normalized distance to target (smooth in [0,1]).
    """

    name = "hard"

    def score(self, servers: list[Server], ctx: dict[str, Any]) -> float:
        target = float(ctx.get("target_cost_performance_ratio", 0.0))
        if target <= 0.0:
            return 1.0

        active = [s for s in servers if s.active]
        total_cost = sum(s.hourly_cost_usd for s in active)
        total_perf = sum(s.performance_units for s in active)
        if total_perf <= 1e-9:
            return 0.0

        current = total_cost / total_perf
        rel_err = abs(current - target) / target
        # Full credit within 5% relative error; linear decay to 0 by ~35% error
        return self._clamp01(1.0 - rel_err / 0.35)


# --- Environment -------------------------------------------------------------


class CloudOpsEnvironment(Environment[CloudOpsAction, CloudOpsObservation, State]):
    """
    Virtual fleet ops: cost, security (SSH exposure), and rightsizing.

    Rewards are incremental improvements in summed grader achievements, clipped to [0, 1].
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS_PER_EPISODE: int = 120

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._servers: list[Server] = []
        self._rng = random.Random()
        self._ctx: dict[str, Any] = {}
        self._prev_grader_scores: dict[str, float] = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        self._graders: tuple[TaskGrader, ...] = (
            EasyIdleTerminationGrader(),
            MediumSSHGrader(),
            HardCostPerformanceGrader(),
        )

    def _snapshot_servers(self) -> list[Server]:
        return [s.model_copy(deep=True) for s in self._servers]

    def _fleet_ratio(self) -> float:
        active = [s for s in self._servers if s.active]
        total_cost = sum(s.hourly_cost_usd for s in active)
        total_perf = sum(s.performance_units for s in active)
        if total_perf <= 1e-9:
            return 0.0
        return total_cost / total_perf

    def _compute_grader_scores(self) -> dict[str, float]:
        return {g.name: float(g.score(self._servers, self._ctx)) for g in self._graders}

    def _incremental_reward(self, current: dict[str, float]) -> float:
        delta = 0.0
        for k in ("easy", "medium", "hard"):
            prev = self._prev_grader_scores.get(k, 0.0)
            cur = current.get(k, 0.0)
            delta += max(0.0, cur - prev)
        self._prev_grader_scores = dict(current)
        return float(min(1.0, delta))

    def _all_tasks_solved(self, scores: dict[str, float]) -> bool:
        return all(scores.get(k, 0.0) >= 0.999 for k in ("easy", "medium", "hard"))

    def _build_observation(
        self,
        done: bool,
        reward: float,
        feedback: str,
    ) -> CloudOpsObservation:
        scores = self._compute_grader_scores()
        return CloudOpsObservation(
            summary_message=feedback,
            servers=self._snapshot_servers(),
            current_cost_performance_ratio=self._fleet_ratio(),
            target_cost_performance_ratio=float(
                self._ctx.get("target_cost_performance_ratio", 0.0)
            ),
            grader_scores=scores,
            done=done,
            reward=reward,
            metadata={
                "action_feedback": feedback,
                "grader_scores": scores,
                "step_count": self._state.step_count,
            },
        )

    def _spawn_episode(self, seed: Optional[int]) -> None:
        self._rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        sid = lambda i: f"srv-{i:02d}"

        # Baseline fleet
        servers: list[Server] = [
            Server(
                id=sid(1),
                cpu_utilization_percent=62.0,
                hourly_cost_usd=0.06,
                security_status=SecurityStatus.SECURE,
                performance_units=42.0,
                active=True,
                ssh_listens_on="10.0.0.0/8",
            ),
            Server(
                id=sid(2),
                cpu_utilization_percent=3.0,
                hourly_cost_usd=0.06,
                security_status=SecurityStatus.SECURE,
                performance_units=42.0,
                active=True,
                ssh_listens_on="127.0.0.1",
            ),
            Server(
                id=sid(3),
                cpu_utilization_percent=48.0,
                hourly_cost_usd=0.14,
                security_status=SecurityStatus.SSH_EXPOSED_WORLD,
                performance_units=110.0,
                active=True,
                ssh_listens_on="0.0.0.0",
            ),
        ]

        # Randomize tiers slightly so hard task needs at least one rightsizing step sometimes
        if self._rng.random() < 0.5:
            s = servers[0]
            cost, perf = TIER_SPECS["performance"]
            servers[0] = s.model_copy(
                update={"hourly_cost_usd": cost, "performance_units": perf}
            )

        self._servers = servers

        initial_idle = {
            s.id for s in self._servers if s.active and s.cpu_utilization_percent < 5.0
        }
        initial_vuln = {
            s.id
            for s in self._servers
            if s.active and s.security_status == SecurityStatus.SSH_EXPOSED_WORLD
        }

        # -------------------------
        # Fix A: make HARD reachable exactly.
        #
        # The HARD grader only considers currently-active servers.
        # EASY requires terminating all initially-idle servers, so after
        # EASY=1.0, the set of active servers is deterministic.
        #
        # We sample a discrete tier assignment for that post-EASY active set
        # and set the target ratio to the resulting (reachable) ratio.
        # Then the oracle (and any perfect agent) can hit rel_err==0.0.
        # -------------------------
        post_easy_active = [
            s for s in self._servers if s.active and s.id not in initial_idle
        ]
        tier_assignment_for_target: dict[str, str] = {}
        total_cost = 0.0
        total_perf = 0.0
        for s in post_easy_active:
            tier = self._rng.choice(list(TIER_SPECS.keys()))
            tier_assignment_for_target[s.id] = tier
            cost, perf = TIER_SPECS[tier]
            total_cost += float(cost)
            total_perf += float(perf)

        # If everything was idle (edge-case), treat hard as already solved.
        target_ratio = 0.0 if total_perf <= 1e-12 else (total_cost / total_perf)

        self._ctx = {
            "initial_idle_server_ids": initial_idle,
            "initial_vulnerable_server_ids": initial_vuln,
            "target_cost_performance_ratio": target_ratio,
            "target_tier_assignment": tier_assignment_for_target,
        }
        self._prev_grader_scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}

    def _find_server(self, server_id: str) -> Optional[int]:
        for i, s in enumerate(self._servers):
            if s.id == server_id:
                return i
        return None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CloudOpsObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._spawn_episode(seed)
        feedback = (
            "Episode started. Tasks: (easy) terminate an idle server (CPU < 5%); "
            "(medium) close SSH exposure on 0.0.0.0; "
            "(hard) match target cost/performance ratio via set_instance_tier."
        )
        return self._build_observation(done=False, reward=0.0, feedback=feedback)

    def step(
        self,
        action: CloudOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CloudOpsObservation:
        self._state.step_count += 1
        feedback = ""

        idx = self._find_server(action.server_id) if action.server_id else None

        if action.command == "noop":
            feedback = "No operation performed."
        elif action.command == "terminate_server":
            if idx is None:
                feedback = f"Unknown server_id {action.server_id!r}."
            else:
                s = self._servers[idx]
                if not s.active:
                    feedback = f"Server {s.id} already terminated."
                elif s.cpu_utilization_percent >= 5.0:
                    feedback = (
                        f"Refused terminate: {s.id} is not idle (CPU "
                        f"{s.cpu_utilization_percent:.1f}% >= 5%)."
                    )
                else:
                    self._servers[idx] = s.model_copy(update={"active": False})
                    feedback = f"Terminated idle server {s.id}."
        elif action.command == "fix_ssh_exposure":
            if idx is None:
                feedback = f"Unknown server_id {action.server_id!r}."
            else:
                s = self._servers[idx]
                if not s.active:
                    feedback = f"Server {s.id} is terminated; nothing to fix."
                elif s.security_status != SecurityStatus.SSH_EXPOSED_WORLD:
                    feedback = f"{s.id} does not have world-exposed SSH."
                else:
                    self._servers[idx] = s.model_copy(
                        update={
                            "security_status": SecurityStatus.SECURE,
                            "ssh_listens_on": "10.0.0.0/8",
                        }
                    )
                    feedback = f"Restricted SSH on {s.id} to private ranges."
        elif action.command == "set_instance_tier":
            if idx is None:
                feedback = f"Unknown server_id {action.server_id!r}."
            elif action.instance_tier not in TIER_SPECS:
                feedback = f"Invalid instance_tier {action.instance_tier!r}."
            else:
                s = self._servers[idx]
                if not s.active:
                    feedback = f"Server {s.id} is terminated; cannot resize."
                else:
                    cost, perf = TIER_SPECS[action.instance_tier]
                    self._servers[idx] = s.model_copy(
                        update={
                            "hourly_cost_usd": cost,
                            "performance_units": perf,
                        }
                    )
                    feedback = (
                        f"Set {s.id} to tier {action.instance_tier!r} "
                        f"(cost={cost}, perf_units={perf})."
                    )
        else:
            feedback = f"Unsupported command {action.command!r}."

        scores = self._compute_grader_scores()
        inc_reward = self._incremental_reward(scores)

        done = self._all_tasks_solved(scores) or (
            self._state.step_count >= self.MAX_STEPS_PER_EPISODE
        )

        return self._build_observation(done=done, reward=inc_reward, feedback=feedback)

    @property
    def state(self) -> State:
        return self._state
