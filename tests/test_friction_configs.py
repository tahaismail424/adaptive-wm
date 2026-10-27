"""v0.5 friction experiment (stock / slippery / mixed checkpoints, each evaluated
on the stock and slippery val demos): train + eval configs, normalization stats
taken from each checkpoint's own training file, and the env playing every val
episode on its own floor."""

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from conftest import ROOT, STABLEWM_HOME
from eval import sample_eval_starts, stats_dataset_name

FRICTION = STABLEWM_HOME / "datasets" / "friction_pusht"
CONDITIONS = ["stock", "slippery", "mixed"]
FLOORS = {"stock": {"sim.damping": 0.0, "sim.floor_friction": 0.0},
          "slippery": {"sim.damping": 1.0, "sim.floor_friction": 0.1}}
ENV_ID = {"stock": 0, "slippery": 1}


def compose(kind, name, overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config" / kind), version_base=None):
        return hydra_compose(name, overrides=list(overrides))


def complete(cond, split):
    """The export's h5 file exists and has all the rows its manifest promises
    (files may still be uploading)."""
    manifest = FRICTION / cond / f"pusht_v05_friction_{cond}.manifest.yaml"
    path = FRICTION / cond / f"pusht_v05_friction_{cond}_{split}.h5"
    if not (manifest.exists() and path.exists()):
        return False
    rows = yaml.safe_load(manifest.read_text())["splits"][split]["rows"]
    try:
        with h5py.File(path, "r") as f:
            return int(f["ep_len"][:].sum()) == rows
    except (OSError, KeyError):
        return False


# ---------------------------------------------------------------- train configs


@pytest.mark.parametrize("cond", CONDITIONS)
def test_train_config(cond):
    cfg = compose("train", "lewm", [f"data=pusht_friction_{cond}"])
    stem = f"friction_pusht/{cond}/pusht_v05_friction_{cond}"
    assert cfg.data.dataset.name == f"{stem}_train.h5"
    assert cfg.data.val_set == f"{stem}_val.h5"
    assert cfg.data.dataset.frameskip == 5 and cfg.data.dataset.num_steps == 4
    assert cfg.model._target_ == "jepa.JEPA"  # plain LeWM


@pytest.mark.parametrize("cond", CONDITIONS)
def test_train_datasets_build(cond):
    """train.build_datasets on the real files, once both are fully uploaded."""
    if not (complete(cond, "train") and complete(cond, "val")):
        pytest.skip(f"{cond} export not (fully) present yet")
    from train import build_datasets

    cfg = compose("train", "lewm", [f"data=pusht_friction_{cond}"])
    train_set, val_set = build_datasets(cfg, torch.Generator().manual_seed(0))
    assert len(train_set) > 0 and 0 < len(val_set) < 0.25 * len(train_set)
    assert cfg.model.action_encoder.input_dim == 10
    sample = val_set[0]
    assert sample["pixels"].shape == (4, 3, 224, 224)
    with h5py.File(FRICTION / cond / f"pusht_v05_friction_{cond}_train.h5") as f:
        train_parents = set(f["parent_idx"][:][f["ep_offset"][:]].tolist())
        train_envs = set(f["env_id"][:][f["ep_offset"][:]].tolist())
    with h5py.File(FRICTION / cond / f"pusht_v05_friction_{cond}_val.h5") as f:
        val_parents = set(f["parent_idx"][:][f["ep_offset"][:]].tolist())
    assert train_parents.isdisjoint(val_parents)
    assert train_envs == ({0, 1} if cond == "mixed" else {ENV_ID[cond]})


# ---------------------------------------------------------------- normalization stats


def write_run(home, run, dataset_name):
    run_dir = home / "checkpoints" / run
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"data": {"dataset": {"name": dataset_name}}}), run_dir / "config.yaml")
    (run_dir / "weights_epoch_10.pt").touch()


def test_stats_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    write_run(tmp_path, "lewm_friction_slippery", "friction_pusht/slippery/pusht_v05_friction_slippery_train.h5")
    cfg = OmegaConf.create({"policy": "lewm_friction_slippery/weights_epoch_10.pt",
                            "train": {"dataset_name": "from_checkpoint"}})
    assert stats_dataset_name(cfg) == "friction_pusht/slippery/pusht_v05_friction_slippery_train"
    cfg.policy = "lewm_friction_slippery/"  # folder form
    assert stats_dataset_name(cfg) == "friction_pusht/slippery/pusht_v05_friction_slippery_train"


def test_stats_explicit_none_and_random():
    assert stats_dataset_name(OmegaConf.create({"train": {"dataset_name": "x/y_train"}})) == "x/y_train"
    assert stats_dataset_name(OmegaConf.create({"policy": "a/b.pt"})) is None
    random = OmegaConf.create({"policy": "random", "train": {"dataset_name": "from_checkpoint"}})
    assert stats_dataset_name(random) is None


def test_stats_from_checkpoint_without_config_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    cfg = OmegaConf.create({"policy": "nope/weights.pt", "train": {"dataset_name": "from_checkpoint"}})
    with pytest.raises(ValueError, match="config.yaml"):
        stats_dataset_name(cfg)


@pytest.mark.skipif(not (STABLEWM_HOME / "checkpoints/lewm_pusht_shapeI/config.yaml").exists(),
                    reason="shapeI run missing")
def test_stats_from_a_real_run():
    cfg = OmegaConf.create({"policy": "lewm_pusht_shapeI/weights_epoch_30.pt",
                            "train": {"dataset_name": "from_checkpoint"}})
    assert stats_dataset_name(cfg) == "varied_pusht/pusht_v0_only_shape_I_train"


# ---------------------------------------------------------------- eval configs


@pytest.mark.parametrize("floor", ["stock", "slippery"])
def test_eval_config(floor):
    cfg = compose("eval", f"pusht_friction_{floor}_val")
    assert cfg.world.env_name == "pusht_scenarios.env:pts/PushTBounded-v0"
    assert not cfg.world.get("render_contacts", False)  # pusht-scenarios frames have no contact dots
    assert cfg.eval.get("callables") is None and not cfg.eval.get("variation_overrides")
    assert cfg.eval.dataset_name == f"friction_pusht/{floor}/pusht_v05_friction_{floor}_val"
    assert cfg.eval.only_original and cfg.eval.per_env == 100 and not cfg.eval.allow_fewer
    assert cfg.train.dataset_name == "from_checkpoint"
    assert (cfg.eval.goal_offset_steps, cfg.eval.eval_budget) == (25, 50)
    assert cfg.solver.batch_size == cfg.eval.chunk_size


def test_eval_configs_differ_only_in_data():
    a, b = compose("eval", "pusht_friction_stock_val"), compose("eval", "pusht_friction_slippery_val")
    for key in ("world", "plan_config", "solver", "seed", "dataset", "train"):
        assert a[key] == b[key], key
    for key in set(a.eval) - {"dataset_name"}:
        assert a.eval[key] == b.eval[key], key


@pytest.mark.parametrize("floor", ["stock", "slippery"])
def test_val_sampling(floor):
    if not complete(floor, "val"):
        pytest.skip(f"{floor} val not present yet")
    import stable_worldmodel as swm

    cfg = compose("eval", f"pusht_friction_{floor}_val")
    ds = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=["state"], cache_dir=STABLEWM_HOME)
    rows, episodes, starts = sample_eval_starts(
        ds, cfg.eval.num_eval, cfg.eval.goal_offset_steps, cfg.seed, per_env=cfg.eval.per_env,
        only_original=True, allow_fewer=False,
    )
    data = ds.get_row_data(rows)
    assert len(rows) == 100 and np.asarray(data["is_original"]).astype(bool).all()
    assert set(np.asarray(data["env_id"]).reshape(-1).tolist()) == {ENV_ID[floor]}
    assert len(set(episodes.tolist())) > 10  # spread over the held-out demos


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("floor", ["stock", "slippery"])
def test_eval_run_plays_each_floor(floor, tmp_path, monkeypatch):
    """eval.run with a random policy on 2 val starts: every env is reset onto
    the val episode's own floor (damping/friction from the variation.* columns)."""
    if not complete(floor, "val"):
        pytest.skip(f"{floor} val not present yet")
    import eval as eval_mod
    from pusht_scenarios.env import variations

    (tmp_path / "datasets").symlink_to(STABLEWM_HOME / "datasets")
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    cfg = compose("eval", f"pusht_friction_{floor}_val", [
        "policy=random", "eval.per_env=2", "eval.chunk_size=2", "eval.eval_budget=4",
        "plan_config.horizon=1", "plan_config.action_block=1", "eval.video=none",
    ])
    built, world_cls = [], eval_mod.swm.World

    def recording_world(*args, **kwargs):
        built.append(world_cls(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(eval_mod.swm, "World", recording_world)
    eval_mod.run(cfg)

    assert len(built) == 1 and len(built[0].envs.envs) == 2
    for env in built[0].envs.envs:
        flat = variations.flatten(env)
        for key, value in FLOORS[floor].items():
            assert flat[key] == pytest.approx(value), key
    out = tmp_path / "random" / "pusht_scenarios.env_pts_PushTBounded-v0"
    results = json.loads((out / f"pusht_friction_{floor}_val_results_per_env.json").read_text())
    assert results["metrics"]["num_episodes"] == 2
