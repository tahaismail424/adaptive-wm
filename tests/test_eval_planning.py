"""The trained-model planning path of eval.py: solver construction from the
eval configs and a short CPU planning run with a real checkpoint. (Random-policy
evals never build a solver, which is how the `model=` -> `cost=` API change in
swm's solvers went unnoticed until a GPU eval crashed.)"""

import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from conftest import ROOT, STABLEWM_HOME, VARIED_PUSHT
from eval import (
    build_policy, evaluate_in_chunks, fit_scalers, get_dataset, img_transform, make_chunks,
    sample_eval_starts,
)

CHECKPOINT = "lewm_pusht_shapeI/weights_epoch_5.pt"


def compose_eval(name, overrides=()):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        return compose(name, overrides=list(overrides))


class StubCost(torch.nn.Module):
    """Minimal Costable: get_cost(info_dict, candidates) -> (B, S)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def criterion(self, info_dict, action_candidates):
        return action_candidates.pow(2).sum(dim=(-1, -2))

    def get_cost(self, info_dict, action_candidates):
        self.calls += 1
        return self.criterion(info_dict, action_candidates)


@pytest.mark.parametrize("solver", ["cem", "adam"])
def test_solver_configs_take_a_cost(solver):
    cfg = compose_eval("pusht_shapeI", [f"solver={solver}", "solver.device=cpu"])
    assert "model" not in cfg.solver
    stub = StubCost()
    built = instantiate(cfg.solver, cost=stub)
    assert built.cost is stub


@pytest.mark.parametrize("name", ["pusht_shapeI", "pusht_shapeI_val", "pusht_night", "pusht_heavy"])
def test_solver_target_matches_varied_configs(name):
    cfg = compose_eval(name, ["solver.device=cpu"])
    built = instantiate(cfg.solver, cost=StubCost())
    assert type(built).__name__ == "CEMSolver"
    assert built.batch_size == cfg.eval.chunk_size  # a whole chunk planned per GPU batch


def test_random_policy_needs_no_solver():
    cfg = compose_eval("pusht_shapeI")
    assert isinstance(build_policy(cfg, process={}, transform={}), swm.policy.RandomPolicy)


@pytest.mark.skipif(
    not (STABLEWM_HOME / "checkpoints" / CHECKPOINT).exists()
    or not (VARIED_PUSHT / "pusht_v0_only_shape_I_val.h5").exists(),
    reason="shape_I checkpoint or val export missing",
)
def test_trained_policy_plans_on_cpu():
    """eval.py's own build_policy + evaluate_in_chunks with a real checkpoint,
    a tiny CEM budget, on CPU: everything a GPU eval job does, just smaller."""
    cfg = compose_eval("pusht_shapeI_val", [
        f"policy={CHECKPOINT}", "solver.device=cpu", "solver.num_samples=8",
        "solver.n_steps=2", "solver.topk=4", "eval.chunk_size=2",
    ])
    cfg.world.max_episode_steps = 12

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats = get_dataset(cfg, OmegaConf.select(cfg, "train.dataset_name"))
    process = fit_scalers(stats, cfg.dataset.keys_to_cache)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    policy = build_policy(cfg, process, transform)
    assert isinstance(policy, swm.policy.WorldModelPolicy)
    model = policy.solver.cost
    assert type(model).__name__ == "JEPA" and not model.training
    assert next(model.parameters()).device.type == "cpu"

    calls = {"n": 0}
    get_cost = model.get_cost

    def counting_get_cost(*args, **kwargs):
        calls["n"] += 1
        return get_cost(*args, **kwargs)

    model.get_cost = counting_get_cost

    rows, episodes, starts = sample_eval_starts(
        dataset, 0, cfg.eval.goal_offset_steps, cfg.seed, per_env=1,
        only_original=True, allow_fewer=True,
    )
    keep = slice(0, 2)

    def make_world(n):
        world = swm.World(**{**cfg.world, "num_envs": n}, image_shape=(224, 224))
        world.set_policy(policy)
        return world

    chunks = make_chunks(np.zeros(2, dtype=bool), chunk_size=cfg.eval.chunk_size)
    successes, videos = evaluate_in_chunks(
        make_world, dataset, episodes[keep], starts[keep], chunks,
        goal_offset=cfg.eval.goal_offset_steps, eval_budget=6,
    )
    assert successes.shape == (2,) and videos == []
    assert calls["n"] > 0  # the planner really queried the world model
