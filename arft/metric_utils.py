from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.trainer.ppo.metric_utils import compute_data_metrics as compute_sequence_data_metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes metrics for agent_flow, aggregating step-level metrics into trajectory-level metrics.
    """
    # 1. Compute standard sequence-level metrics (each sequence is a step in agent_flow).
    # veRL historically places scores, rewards, advantages, and returns under the
    # ``critic/`` namespace even when no critic is used.  Keep that namespace for
    # actual value-function diagnostics only.
    raw_metrics = compute_sequence_data_metrics(batch, use_critic=use_critic)
    metric_prefixes = {
        "critic/score/": "train/scores/",
        "critic/rewards/": "train/rewards/",
        "critic/advantages/": "train/advantages/",
        "critic/returns/": "train/returns/",
    }
    metrics = {}
    for key, value in raw_metrics.items():
        for old_prefix, new_prefix in metric_prefixes.items():
            if key.startswith(old_prefix):
                key = new_prefix + key.removeprefix(old_prefix)
                break
        metrics[key] = value

    if "trajectory_uids" not in batch.non_tensor_batch:
        return metrics

    # 2. Extract trajectory information
    trajectory_uids = batch.non_tensor_batch["trajectory_uids"]
    unique_uids, trajectory_indices = np.unique(trajectory_uids, return_inverse=True)
    num_trajectories = len(unique_uids)
    trajectory_indices_torch = torch.from_numpy(trajectory_indices).to(batch.batch["responses"].device)

    # 3. Aggregate metrics to trajectory level
    response_info = _compute_response_info(batch)
    step_prompt_lengths = response_info["prompt_length"]
    step_response_lengths = response_info["response_length"]

    # Trajectory Prompt Length (Sum of steps)
    traj_prompt_lengths = torch.zeros(num_trajectories, device=step_prompt_lengths.device)
    traj_prompt_lengths.scatter_add_(0, trajectory_indices_torch, step_prompt_lengths)

    # Trajectory Response Length (Sum of steps)
    traj_response_lengths = torch.zeros(num_trajectories, device=step_response_lengths.device)
    traj_response_lengths.scatter_add_(0, trajectory_indices_torch, step_response_lengths)

    # Trajectory Num Steps
    traj_num_steps = torch.zeros(num_trajectories, device=step_response_lengths.device)
    traj_num_steps.scatter_add_(0, trajectory_indices_torch, torch.ones_like(step_response_lengths))

    # Trajectory Score/Reward (Sum of step scores/rewards)
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    traj_scores = torch.zeros(num_trajectories, device=sequence_score.device)
    traj_scores.scatter_add_(0, trajectory_indices_torch, sequence_score)

    traj_rewards = torch.zeros(num_trajectories, device=sequence_reward.device)
    traj_rewards.scatter_add_(0, trajectory_indices_torch, sequence_reward)

    # Add trajectory-level metrics
    metrics.update(
        {
            # Trajectory-level prompt length
            "prompt_length/trajectory/mean": traj_prompt_lengths.mean().item(),
            "prompt_length/trajectory/max": traj_prompt_lengths.max().item(),
            "prompt_length/trajectory/min": traj_prompt_lengths.min().item(),
            # Trajectory-level response length
            "response_length/trajectory/mean": traj_response_lengths.mean().item(),
            "response_length/trajectory/max": traj_response_lengths.max().item(),
            "response_length/trajectory/min": traj_response_lengths.min().item(),
            # Trajectory-level num steps
            "num_steps/mean": traj_num_steps.mean().item(),
            "num_steps/max": traj_num_steps.max().item(),
            "num_steps/min": traj_num_steps.min().item(),
            # Trajectory-level score/reward
            "train/scores/trajectory/mean": traj_scores.mean().item(),
            "train/scores/trajectory/max": traj_scores.max().item(),
            "train/scores/trajectory/min": traj_scores.min().item(),
            "train/rewards/trajectory/mean": traj_rewards.mean().item(),
            "train/rewards/trajectory/max": traj_rewards.max().item(),
            "train/rewards/trajectory/min": traj_rewards.min().item(),
        }
    )

    return metrics
