"""eval.py helpers (scalers, start sampling, per-env breakdown), the varied
Push-T eval configs, and swm's dataset eval restoring held-out envs."""

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import yaml
from omegaconf import OmegaConf

from conftest import ROOT, STABLEWM_HOME, VARIED_PUSHT, write_h5
from eval import fit_scalers, load_manifest, per_env_results, sample_eval_starts

EVAL_CONFIGS = ["pusht_shapeI", "pusht_night", "pusht_heavy"]


def h5(root, name):
    return swm.data.HDF5Dataset(name, keys_to_cache=["action", "proprio", "state"], cache_dir=root)


@pytest.fixture
def eval_home(tmp_path):
    """syn/x_train.h5 (~+10) and syn/x_test_eval.h5 (~-10, 3 envs) + manifest."""
    data = tmp_path / "datasets" / "syn"
    write_h5(data / "x_train.h5", [40, 30], state_value=10.0)
    ep_lens = [30, 40, 28, 35, 50, 20]  # last one is too short for a 25-step goal
    write_h5(data / "x_test_eval.h5", ep_lens, state_value=-10.0)
    env_of_episode = [7, 7, 8, 8, 9, 9]
    with h5py.File(data / "x_test_eval.h5", "a") as f:
        f.create_dataset("env_id", data=np.repeat(env_of_episode, ep_lens))
    manifest = {"envs": [
        {"env_id": 7, "name": "a", "levels": {"mass": "heavy", "shape": "T"}},
        {"env_id": 8, "name": "b", "levels": {"mass": "heavy", "shape": "I"}},
        {"env_id": 9, "name": "c", "levels": {"mass": "light", "shape": "I"}},
    ]}
    (data / "x.manifest.yaml").write_text(yaml.safe_dump(manifest))
    return tmp_path


# ---------------------------------------------------------------- helpers


def test_scalers_use_the_stats_dataset(eval_home):
    train, test = h5(eval_home, "syn/x_train"), h5(eval_home, "syn/x_test_eval")
    process = fit_scalers(train, ["pixels", "action", "proprio", "state"])
    assert set(process) == {"action", "proprio", "state", "goal_proprio", "goal_state"}
    assert process["goal_state"] is process["state"]
    state = train.get_col_data("state")
    np.testing.assert_allclose(process["state"].mean_, state.mean(0))
    assert not np.allclose(process["state"].mean_, test.get_col_data("state").mean(0))


def test_sampled_starts_leave_room_for_the_goal(eval_home):
    ds = h5(eval_home, "syn/x_test_eval")
    rows, episodes, starts = sample_eval_starts(ds, num_eval=40, goal_offset=25, seed=0)
    assert len(rows) == 40 and np.all(np.diff(rows) > 0)
    lengths = {e: n for e, n in enumerate([30, 40, 28, 35, 50, 20])}
    for ep, start in zip(episodes, starts):
        assert start + 25 < lengths[int(ep)]
    assert 5 not in set(episodes.tolist())  # 20-step episode never qualifies
    again = sample_eval_starts(ds, num_eval=40, goal_offset=25, seed=0)
    np.testing.assert_array_equal(rows, again[0])


def test_too_few_starts_raises(eval_home):
    with pytest.raises(ValueError, match="Not enough"):
        sample_eval_starts(h5(eval_home, "syn/x_test_eval"), num_eval=500, goal_offset=25, seed=0)


def test_per_env_breakdown(eval_home):
    ds = h5(eval_home, "syn/x_test_eval")
    rows, episodes, _ = sample_eval_starts(ds, num_eval=30, goal_offset=25, seed=1)
    env_of_episode = np.array([7, 7, 8, 8, 9, 9])
    successes = env_of_episode[episodes] == 8  # pretend only env 8 succeeds
    manifest = load_manifest(ds)
    assert manifest is not None

    res = per_env_results(ds, rows, successes, manifest)
    assert set(res["by_env"]) == set(env_of_episode[episodes].tolist())
    assert res["by_env"][8]["success_rate"] == 100.0
    assert res["by_env"][8]["levels"] == {"mass": "heavy", "shape": "I"}
    assert sum(r["episodes"] for r in res["by_env"].values()) == 30
    for env_id in (7, 9):
        assert res["by_env"][env_id]["successes"] == 0
    n8 = res["by_env"][8]["episodes"]
    assert res["by_factor"]["shape"]["I"]["successes"] == n8
    assert res["by_factor"]["shape"]["T"]["successes"] == 0


def test_per_env_breakdown_without_env_id(synthetic_home):
    ds = h5(synthetic_home["root"], "syn/val")
    assert per_env_results(ds, np.array([0, 1]), np.array([True, False])) is None


# ---------------------------------------------------------------- configs


def compose_eval(name):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        return compose(name)


@pytest.mark.parametrize("name", EVAL_CONFIGS)
def test_eval_config(name):
    import importlib

    import gymnasium as gym
    from hydra.utils import get_class

    cfg = compose_eval(name)
    assert cfg.world.env_name == "pusht_scenarios.env:pts/PushTBounded-v0"
    # gym.make (what swm.World uses) imports the "module:" prefix; gym.spec doesn't
    module, env_id = cfg.world.env_name.split(":")
    importlib.import_module(module)
    gym.spec(env_id)
    get_class(cfg.solver._target_)  # stable_worldmodel.planning.CEMSolver at the pinned swm
    assert cfg.eval.get("callables") is None
    eval_name, train_name = cfg.eval.dataset_name, cfg.train.dataset_name
    assert eval_name.startswith("varied_pusht/") and eval_name.endswith("_test_eval")
    assert train_name == eval_name.replace("_test_eval", "_train")
    assert OmegaConf.select(cfg, "dataset.stats") is None


@pytest.mark.skipif(not VARIED_PUSHT.exists(), reason=f"{VARIED_PUSHT} not present")
@pytest.mark.parametrize("name", EVAL_CONFIGS)
def test_eval_config_files_exist(name):
    cfg = compose_eval(name)
    for ds in (cfg.eval.dataset_name, cfg.train.dataset_name):
        assert (STABLEWM_HOME / "datasets" / f"{ds}.h5").exists(), ds


# ---------------------------------------------------------------- integration


@pytest.mark.skipif(
    not (VARIED_PUSHT / "pusht_v0_only_shape_I_test_eval.h5").exists(), reason="shape_I export missing"
)
def test_dataset_eval_restores_held_out_envs():
    """swm's World.evaluate on the shape_I test file must rebuild each episode's
    env (shape I, its mass/friction/colors) via reset_options_from_dataset, not
    run default Push-T. Random policy, tiny budget, CPU only."""
    from pusht_scenarios.env import variations

    ds = swm.data.HDF5Dataset(
        "varied_pusht/pusht_v0_only_shape_I_test_eval", keys_to_cache=["state"],
        cache_dir=STABLEWM_HOME,
    )
    rows, episodes, starts = sample_eval_starts(ds, num_eval=3, goal_offset=25, seed=0)
    world = swm.World(
        env_name="pusht_scenarios.env:pts/PushTBounded-v0", num_envs=3,
        max_episode_steps=10, image_shape=(224, 224),
    )
    world.set_policy(swm.policy.RandomPolicy())
    metrics = world.evaluate(
        dataset=ds, start_steps=starts.tolist(), goal_offset=25, eval_budget=2,
        episodes_idx=episodes.tolist(),
    )
    assert len(metrics["episode_successes"]) == 3

    row_data = ds.get_row_data(rows)
    for i, env in enumerate(world.envs.envs):
        flat = variations.flatten(env)
        assert int(flat["block.shape"]) == 5  # 'I'
        for key in ("sim.block_mass", "sim.damping", "sim.floor_friction", "block.color"):
            np.testing.assert_allclose(flat[key], row_data[f"variation.{key}"][i])

    res = per_env_results(ds, rows, metrics["episode_successes"], load_manifest(ds))
    for env_id, r in res["by_env"].items():
        assert r["levels"]["shape"] == "I" and env_id >= 54  # test envs follow the 54 train ids
