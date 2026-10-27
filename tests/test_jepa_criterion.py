"""JEPA.criterion (the planning cost) with several envs per solver batch."""

import torch
import torch.nn.functional as F

from jepa import JEPA


def reference_cost(pred_emb, goal_emb):
    """LeWM's original computation, which is only valid for one env (B == 1)."""
    goal = goal_emb[..., -1:, :].expand_as(pred_emb)
    return F.mse_loss(pred_emb[..., -1:, :], goal[..., -1:, :], reduction="none").sum(
        dim=tuple(range(2, pred_emb.ndim))
    )


def test_batched_cost_equals_per_env_cost():
    torch.manual_seed(0)
    B, S, T, D = 4, 7, 6, 16
    pred = torch.randn(B, S, T, D)
    goal = torch.randn(B, 3, D)  # get_cost encodes the goal frames: (B, T_goal, D)

    batched = JEPA.criterion(None, {"predicted_emb": pred, "goal_emb": goal})
    assert batched.shape == (B, S)
    for b in range(B):
        one = reference_cost(pred[b : b + 1], goal[b : b + 1])  # what LeWM computed with B == 1
        torch.testing.assert_close(batched[b : b + 1], one)


def test_each_env_is_compared_with_its_own_goal():
    B, S, T, D = 3, 5, 2, 4
    goal = torch.arange(B, dtype=torch.float32)[:, None, None].expand(B, 1, D).clone()
    pred = goal[:, None].expand(B, S, T, D).clone()  # every env predicts exactly its own goal
    cost = JEPA.criterion(None, {"predicted_emb": pred, "goal_emb": goal})
    torch.testing.assert_close(cost, torch.zeros(B, S))


def test_single_env_unchanged():
    torch.manual_seed(1)
    pred, goal = torch.randn(1, 9, 6, 8), torch.randn(1, 3, 8)
    torch.testing.assert_close(
        JEPA.criterion(None, {"predicted_emb": pred, "goal_emb": goal}), reference_cost(pred, goal)
    )
