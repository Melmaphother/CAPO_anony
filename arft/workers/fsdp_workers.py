"""FSDP worker wrappers that load ARFT policy-loss registrations per Ray actor."""

from __future__ import annotations

import numpy as np

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as _VerlAsyncActorRolloutRefWorker


def _uniform_global_step(global_steps) -> int | None:
    """Reduce a sliceable per-row step field to the one driver update it represents."""
    if global_steps is None:
        return None
    values = np.asarray(global_steps).reshape(-1)
    if values.size == 0:
        return None
    step = int(values[0])
    if not np.all(values == step):
        raise ValueError("Actor update received rows from multiple training global steps.")
    return step


class AsyncActorRolloutRefWorker(_VerlAsyncActorRolloutRefWorker):
    """Upstream FSDP worker with ARFT-only policy-loss registration."""

    def __init__(self, *args, **kwargs):
        # Ray actors are independent processes; registration in the driver alone
        # would not make the custom loss mode visible here.
        import arft.policy_losses  # noqa: F401

        super().__init__(*args, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data):
        import arft.policy_losses

        arft.policy_losses.set_global_step(_uniform_global_step(data.non_tensor_batch.get("global_step")))
        return super().update_actor(data)
