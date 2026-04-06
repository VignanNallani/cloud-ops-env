# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloud Ops & Security Auditor environment."""

from typing import Any

from .env import CloudOpsEnvironment
from .models import CloudOpsAction, CloudOpsObservation, SecurityStatus, Server

__all__ = [
    "CloudOpsAction",
    "CloudOpsObservation",
    "CloudOpsEnvironment",
    "CloudOpsEnv",
    "SecurityStatus",
    "Server",
]


def __getattr__(name: str) -> Any:
    if name == "CloudOpsEnv":
        from .client import CloudOpsEnv as _CloudOpsEnv

        return _CloudOpsEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
