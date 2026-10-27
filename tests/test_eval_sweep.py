"""eval.variation_overrides / eval.sweep: forcing physics on top of each eval
episode's restored state (stress tests of a trained model, e.g. block mass and
floor friction on the original Push-T tasks)."""

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

import pusht_scenarios.env  # noqa: F401  (registers pts/*)
from conftest import ROOT, STABLEWM_HOME
from eval import apply_variation_overrides, format_sweep, sweep_points
from pusht_scenarios.env import variations

PUSHT = STABLEWM_HOME / "datasets" / "pusht_expert_train.h5"


def compose_eval(name, overrides=()):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        return compose(name, overrides=list(overrides))


def fake_world(n=2, env_id="pts/PushT-v0"):
    return SimpleNamespace(envs=SimpleNamespace(envs=[gym.make(env_id) for _ in range(n)]))


def rows():
    """A dataset-style init/goal row pair, as swm's dataset eval passes them."""
    state = np.array([150.0, 150.0, 300.0, 300.0, 0.5, 0.0, 0.0])
    goal = np.array([256.0, 256.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0])
    return {"state": state, "variation.sim.block_mass": np.float64(1.0)}, {"goal_state": goal}


# ---------------------------------------------------------------- sweep points


def test_no_sweep_is_one_unnamed_point():
    cfg = OmegaConf.create({"eval": {"variation_overrides": {"sim.block_mass": 3.0}}})
    assert sweep_points(cfg) == [(None, {"sim.block_mass": 3.0})]
    assert sweep_points(OmegaConf.create({"eval": {}})) == [(None, {})]


@pytest.mark.parametrize("name", [
    "pusht", "pusht_shapeI", "pusht_shapeI_val", "pusht_night", "pusht_heavy_val",
])
def test_regular_configs_have_one_plain_point(name):
    """Configs without variation_overrides/sweep (every non-sweep eval) must run
    exactly as before: one unnamed point, no overrides."""
    assert sweep_points(compose_eval(name)) == [(None, {})]


def test_sweep_points_layer_on_base():
    cfg = OmegaConf.create({"eval": {
        "variation_overrides": {"sim.agent_max_force": 20000.0, "sim.block_mass": 1.0},
        "sweep": [{"name": "a", "overrides": {"sim.block_mass": 5.0}}, {"name": "b"}],
    }})
    assert sweep_points(cfg) == [
        ("a", {"sim.agent_max_force": 20000.0, "sim.block_mass": 5.0}),
        ("b", {"sim.agent_max_force": 20000.0, "sim.block_mass": 1.0}),
    ]


@pytest.mark.parametrize("sweep, match", [
    ([{"overrides": {}}], "needs a name"),
    ([{"name": "x"}, {"name": "x"}], "duplicate"),
])
def test_bad_sweeps(sweep, match):
    with pytest.raises(ValueError, match=match):
        sweep_points(OmegaConf.create({"eval": {"sweep": sweep}}))


def test_mass_sweep_config():
    points = dict(sweep_points(compose_eval("pusht_mass_sweep")))
    assert points["stock"] == {"sim.agent_max_force": 0.0}  # uncapped = original env
    masses = [v["sim.block_mass"] for k, v in points.items() if k != "stock"]
    assert masses == sorted(masses) and masses[0] == 0.1 and masses[-1] == 10.0
    assert all(v["sim.agent_max_force"] == 20000.0 for k, v in points.items() if k != "stock")


def test_friction_sweep_config():
    points = dict(sweep_points(compose_eval("pusht_friction_sweep")))
    assert points["stock"] == {"sim.damping": 0.0}  # no momentum = original env
    assert points["friction_0"] == {"sim.damping": 1.0, "sim.floor_friction": 0.0}
    frictions = [v["sim.floor_friction"] for k, v in points.items() if k != "stock"]
    assert frictions == sorted(frictions) and frictions[-1] == 2.0


@pytest.mark.parametrize("name", ["pusht_mass_sweep", "pusht_friction_sweep"])
def test_sweep_configs_share_the_planning_setup(name):
    cfg, base = compose_eval(name), compose_eval("pusht")
    assert cfg.eval.dataset_name == base.eval.dataset_name == "pusht_expert_train"
    assert cfg.plan_config == base.plan_config and cfg.seed == base.seed
    for key in ("goal_offset_steps", "eval_budget", "img_size"):
        assert cfg.eval[key] == base.eval[key]
    assert cfg.world.render_contacts is True  # LeWM's frames have swm's contact dots
    assert cfg.solver.batch_size == cfg.eval.chunk_size
    for _, overrides in sweep_points(cfg):  # every value is valid for the env
        variations.cast_values(gym.make("pts/PushT-v0"), overrides)


# ---------------------------------------------------------------- overrides reach the env


def test_overrides_win_over_dataset_values_and_apply_on_reset():
    world = apply_variation_overrides(fake_world(), {"sim.block_mass": 7.0, "sim.agent_max_force": 20000.0})
    init, goal = rows()
    for env in world.envs.envs:
        options = env.unwrapped.reset_options_from_dataset(init, goal)
        assert options["variation_values"]["sim.block_mass"] == 7.0  # dataset said 1.0
        env.reset(seed=0, options=options)
        flat = variations.flatten(env)
        assert flat["sim.block_mass"] == 7.0 and flat["sim.agent_max_force"] == 20000.0
        np.testing.assert_allclose(env.unwrapped._get_obs()[:2], init["state"][:2], atol=1.0)


def test_no_overrides_leave_env_untouched():
    world = fake_world(1)
    method = world.envs.envs[0].unwrapped.reset_options_from_dataset
    apply_variation_overrides(world, {})
    assert world.envs.envs[0].unwrapped.reset_options_from_dataset == method


def test_unknown_key_and_wrong_env_raise():
    with pytest.raises(ValueError, match="unknown variation"):
        apply_variation_overrides(fake_world(1), {"sim.not_a_knob": 1.0})
    swm_like = SimpleNamespace(envs=SimpleNamespace(envs=[SimpleNamespace(unwrapped=object())]))
    with pytest.raises(ValueError, match="reset_options_from_dataset"):
        apply_variation_overrides(swm_like, {"sim.block_mass": 2.0})


def test_mass_only_matters_with_capped_strength():
    """Why the mass sweep caps agent strength: with swm's kinematic agent the
    block mass has no effect at all; with a cap, heavier blocks move less."""
    def push(overrides):
        env = gym.make("pts/PushT-v0")
        state = np.array([150.0, 256.0, 230.0, 256.0, 0.0, 0.0, 0.0])
        env.reset(seed=0, options={"state": state, "variation_values": variations.cast_values(env, overrides)})
        start = env.unwrapped._get_obs()[2:4].copy()
        for _ in range(12):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        return float(np.linalg.norm(env.unwrapped._get_obs()[2:4] - start))

    assert push({"sim.block_mass": 1.0}) == pytest.approx(push({"sim.block_mass": 10.0}))
    capped = {"sim.agent_max_force": 20000.0}
    light, heavy = push({**capped, "sim.block_mass": 1.0}), push({**capped, "sim.block_mass": 10.0})
    assert light > 2 * heavy > 0


def test_format_sweep():
    table = format_sweep([
        {"name": "stock", "successes": 9, "num_episodes": 10, "success_rate": 90.0, "overrides": {}},
        {"name": "mass_10", "successes": 1, "num_episodes": 10, "success_rate": 10.0, "overrides": {"sim.block_mass": 10.0}},
    ])
    assert "stock" in table and "90.0%" in table and "mass_10" in table


# ---------------------------------------------------------------- end to end


@pytest.mark.skipif(not PUSHT.exists(), reason="pusht_expert_train.h5 missing")
def test_run_sweep_end_to_end(tmp_path, monkeypatch):
    """eval.run with a random policy, 2 starts, 2 sweep points; results in a temp
    STABLEWM_HOME (datasets symlinked) so nothing is written to the real one."""
    import eval as eval_mod

    (tmp_path / "datasets").symlink_to(STABLEWM_HOME / "datasets")
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    cfg = compose_eval("pusht_mass_sweep", [
        "policy=random", "eval.num_eval=2", "eval.chunk_size=2", "eval.eval_budget=4",
        "plan_config.horizon=1", "plan_config.action_block=1", "eval.video=none",
    ])
    OmegaConf.update(cfg, "eval.sweep", [
        {"name": "stock", "overrides": {"sim.agent_max_force": 0.0}},
        {"name": "mass_10", "overrides": {"sim.block_mass": 10.0}},
    ], force_add=True)

    built = []
    world_cls = eval_mod.swm.World

    def recording_world(*args, **kwargs):
        world = world_cls(*args, **kwargs)
        built.append(world)
        return world

    monkeypatch.setattr(eval_mod.swm, "World", recording_world)
    eval_mod.run(cfg)

    out = tmp_path / "random" / "pusht_scenarios.env_pts_PushT-v0"
    summary = json.loads((out / "pusht_mass_sweep_results_sweep.json").read_text())
    assert [p["name"] for p in summary["points"]] == ["stock", "mass_10"]
    assert all(p["num_episodes"] == 2 for p in summary["points"])
    assert (out / "pusht_mass_sweep_results_stock.txt").exists()
    assert (out / "pusht_mass_sweep_results_mass_10.txt").exists()

    stock, heavy = built  # one World (chunk of 2) per sweep point, in order
    for env in stock.envs.envs:
        assert variations.flatten(env)["sim.agent_max_force"] == 0.0
        assert env.unwrapped.render_contacts
    for env in heavy.envs.envs:
        flat = variations.flatten(env)
        assert flat["sim.block_mass"] == 10.0 and flat["sim.agent_max_force"] == 20000.0
