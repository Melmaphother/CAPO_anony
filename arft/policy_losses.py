"""ARFT-owned policy-loss extensions registered at worker startup."""

from __future__ import annotations

import functools
import numpy as np
import torch

from arft.ratio_diagnostics import get_ratio_diagnostics_writer
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo, register_policy_loss

_global_step: int | np.ndarray | torch.Tensor | None = None


def set_global_step(value: int | np.ndarray | torch.Tensor | None) -> None:
    global _global_step
    _global_step = value


def _masked_ratio_metrics(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Return stable, loss-mode-independent ratio diagnostics.

    The clamp matches PPO's numerical guard.  It is applied only while
    exponentiating; ``policy_log_ratio/*`` remains the unclamped quantity.
    """
    with torch.no_grad():
        mask = response_mask.bool()
        log_ratio = log_prob - old_log_prob
        stable_log_ratio = log_ratio.clamp(min=-20.0, max=20.0)
        ratio = torch.exp(stable_log_ratio)
        values = ratio[mask]
        log_values = log_ratio[mask]
        if values.numel() == 0:
            return {}, {}
        quantiles = torch.quantile(values.float(), torch.tensor([0.01, 0.5, 0.99], device=values.device))
        metrics = {
            "actor/policy_ratio/mean": values.mean().item(),
            "actor/policy_ratio/std": values.std(unbiased=False).item(),
            "actor/policy_ratio/min": values.min().item(),
            "actor/policy_ratio/p01": quantiles[0].item(),
            "actor/policy_ratio/p50": quantiles[1].item(),
            "actor/policy_ratio/p99": quantiles[2].item(),
            "actor/policy_ratio/max": values.max().item(),
            "actor/policy_ratio/clipfrac_0p2": ((values < 0.8) | (values > 1.2)).float().mean().item(),
            "actor/policy_log_ratio/mean": log_values.mean().item(),
            "actor/policy_log_ratio/std": log_values.std(unbiased=False).item(),
            "actor/policy_log_ratio/min": log_values.min().item(),
            "actor/policy_log_ratio/max": log_values.max().item(),
        }
        lengths = mask.sum(dim=-1).clamp(min=1)
        per_row_mask = mask.to(dtype=ratio.dtype)
        per_row = {
            "lengths": lengths,
            "log_ratio_sums": (log_ratio * per_row_mask).sum(dim=-1),
            "ratio_sums": (ratio * per_row_mask).sum(dim=-1),
            "ratio_sq_sums": (ratio.square() * per_row_mask).sum(dim=-1),
            "ratio_mins": ratio.masked_fill(~mask, float("inf")).min(dim=-1).values,
            "ratio_maxs": ratio.masked_fill(~mask, float("-inf")).max(dim=-1).values,
            "clip_fractions": (((ratio < 0.8) | (ratio > 1.2)) & mask).sum(dim=-1) / lengths,
        }
        return metrics, per_row


def _record_ratio_diagnostics(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    metrics, per_row = _masked_ratio_metrics(old_log_prob, log_prob, response_mask)
    writer = get_ratio_diagnostics_writer()
    if writer is not None and per_row:
        mask = response_mask.to(dtype=advantages.dtype)
        step_advantages = (advantages * mask).sum(dim=-1) / per_row["lengths"]
        writer.record(
            lengths=per_row["lengths"],
            log_ratio_sums=per_row["log_ratio_sums"],
            advantages=step_advantages,
            global_step=-1 if _global_step is None else _global_step,
            ratio_sums=per_row["ratio_sums"],
            ratio_sq_sums=per_row["ratio_sq_sums"],
            ratio_mins=per_row["ratio_mins"],
            ratio_maxs=per_row["ratio_maxs"],
            clip_fractions=per_row["clip_fractions"],
        )
    return metrics


@register_policy_loss("gspo_ratio_diag")
def compute_policy_loss_gspo_ratio_diag(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
    global_step: int | np.ndarray | torch.Tensor | None = None,
):
    """Exact GSPO loss plus local sufficient statistics for length-scaling tests.

    The returned loss is delegated to upstream GSPO unchanged.  Diagnostics use
    the same pre-clamp token log-ratio used by GSPO, with padding/tool tokens
    excluded by ``response_mask``.
    """
    pg_loss, pg_metrics = compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    return pg_loss, pg_metrics


def _wrap_registered_policy_losses() -> None:
    """Attach ratio diagnostics to every registered policy loss without changing it."""
    for name, loss_fn in list(core_algos.POLICY_LOSS_REGISTRY.items()):
        if getattr(loss_fn, "_arft_ratio_diagnostics", False):
            continue

        @functools.wraps(loss_fn)
        def wrapped(*args, __loss_fn=loss_fn, **kwargs):
            pg_loss, pg_metrics = __loss_fn(*args, **kwargs)
            old_log_prob = kwargs.get("old_log_prob")
            log_prob = kwargs.get("log_prob")
            advantages = kwargs.get("advantages")
            response_mask = kwargs.get("response_mask")
            if all(isinstance(value, torch.Tensor) for value in (old_log_prob, log_prob, advantages, response_mask)):
                pg_metrics = dict(pg_metrics)
                pg_metrics.update(_record_ratio_diagnostics(old_log_prob, log_prob, advantages, response_mask))
            return pg_loss, pg_metrics

        wrapped._arft_ratio_diagnostics = True
        core_algos.POLICY_LOSS_REGISTRY[name] = wrapped


_wrap_registered_policy_losses()
