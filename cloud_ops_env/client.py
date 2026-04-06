# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloud Ops Env Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CloudOpsAction, CloudOpsObservation


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
