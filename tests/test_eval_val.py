"""In-distribution (val) planning evals: only_original / allow_fewer sampling
and the config/eval/pusht_*_val.yaml configs."""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest
import stable_worldmodel as swm

from conftest import ROOT, STABLEWM_HOME, VARIED_PUSHT, write_h5
from eval import sample_eval_starts

#           episode: 0   1   2   3   4   5
EP_LENS =         [40, 40, 40, 30, 60, 22]
ENV =             [0,  0,  0,  1,  1,  2]
IS_ORIGINAL =     [1,  0,  0,  1,  1,  1]  # 1, 2 are augmented copies of 0
# valid starts for a 25-step goal: 15, 15, 15, 5, 35, 0 (episode 5 is too short)

TAGS = ["shapeI", "night", "heavy"]


@pytest.fixture
def ds(tmp_path):
    path = tmp_path / "datasets" / "syn" / "x_val.h5"
    write_h5(path, EP_LENS, state_value=0.0)
    with h5py.File(path, "a") as f:
        f.create_dataset("env_id", data=np.repeat(ENV, EP_LENS))
        f.create_dataset("is_original", data=np.repeat(IS_ORIGINAL, EP_LENS).astype(np.uint8))
    return swm.data.HDF5Dataset("syn/x_val", keys_to_cache=["state"], cache_dir=tmp_path)


def counts_by_env(episodes):
    envs, counts = np.unique(np.asarray(ENV)[np.asarray(episodes, dtype=int)], return_counts=True)
    return dict(zip(envs.tolist(), counts.tolist()))


def sample(ds, **kw):
    return sample_eval_starts(ds, num_eval=kw.pop("num_eval", 0), goal_offset=25, seed=0, **kw)


# ---------------------------------------------------------------- only_original


def test_only_original_skips_augmented_copies(ds):
    _, episodes, starts = sample(ds, per_env=10, only_original=True, allow_fewer=True)
    assert not {1, 2} & set(episodes.tolist())
    for ep, start in zip(episodes, starts):
        assert start + 25 < EP_LENS[int(ep)]


def test_only_original_uniform_mode(ds):
    _, episodes, _ = sample(ds, num_eval=30, only_original=True)
    assert len(episodes) == 30 and not {1, 2} & set(episodes.tolist())


def test_without_only_original_copies_are_used(ds):
    _, episodes, _ = sample(ds, per_env=40)  # env 0: 40 of the 45 starts spread over episodes 0-2
    assert {1, 2} & set(episodes.tolist())


def test_only_original_needs_the_column(synthetic_home):
    plain = swm.data.HDF5Dataset("syn/val", cache_dir=synthetic_home["root"])
    with pytest.raises(ValueError, match="is_original"):
        sample_eval_starts(plain, num_eval=5, goal_offset=5, seed=0, only_original=True)


# ---------------------------------------------------------------- allow_fewer


def test_allow_fewer_takes_what_exists(ds):
    _, episodes, _ = sample(ds, per_env=20, only_original=True, allow_fewer=True)
    # env 0: one original with 15 starts; env 1: 5 + 35; env 2: none (too short)
    assert counts_by_env(episodes) == {0: 15, 1: 20}


def test_allow_fewer_exact_when_enough(ds):
    _, episodes, _ = sample(ds, per_env=10, only_original=True, allow_fewer=True)
    assert counts_by_env(episodes) == {0: 10, 1: 10}


def test_strict_mode_still_raises(ds):
    with pytest.raises(ValueError, match="env 0"):
        sample(ds, per_env=20, only_original=True)


def test_allow_fewer_is_deterministic(ds):
    a = sample(ds, per_env=20, only_original=True, allow_fewer=True)
    b = sample(ds, per_env=20, only_original=True, allow_fewer=True)
    np.testing.assert_array_equal(a[0], b[0])


# ---------------------------------------------------------------- configs


def compose_eval(name):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config/eval"), version_base=None):
        return compose(name)


@pytest.mark.parametrize("tag", TAGS)
def test_val_config_matches_test_config(tag):
    test, val = compose_eval(f"pusht_{tag}"), compose_eval(f"pusht_{tag}_val")
    assert val.eval.dataset_name == test.eval.dataset_name.replace("_test_eval", "_val")
    assert val.train.dataset_name == test.train.dataset_name  # same scaler stats as training
    assert val.eval.only_original is True and val.eval.allow_fewer is True
    assert test.eval.only_original is False and test.eval.allow_fewer is False
    # results and videos must not overwrite the test eval's
    assert val.output.filename != test.output.filename
    assert Path(val.output.filename).stem != Path(test.output.filename).stem
    # everything that defines the planning problem is identical
    for key in ("world", "plan_config", "solver", "seed", "dataset"):
        assert val[key] == test[key], key
    for key in ("goal_offset_steps", "eval_budget", "img_size", "chunk_size", "video"):
        assert val.eval[key] == test.eval[key], key


# ---------------------------------------------------------------- real data


@pytest.mark.skipif(not VARIED_PUSHT.exists(), reason=f"{VARIED_PUSHT} not present")
@pytest.mark.parametrize("tag", TAGS)
def test_val_sampling_on_real_exports(tag):
    cfg = compose_eval(f"pusht_{tag}_val")
    path = STABLEWM_HOME / "datasets" / f"{cfg.eval.dataset_name}.h5"
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    data = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=["state"], cache_dir=STABLEWM_HOME)
    rows, _, starts = sample_eval_starts(
        data, cfg.eval.num_eval, cfg.eval.goal_offset_steps, cfg.seed, per_env=cfg.eval.per_env,
        only_original=True, allow_fewer=True,
    )
    row_data = data.get_row_data(rows)
    assert np.asarray(row_data["is_original"]).astype(bool).all()

    # expected count per env, straight from the file: min(per_env, valid starts of its originals)
    with h5py.File(path) as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        orig, env = f["is_original"][:][off].astype(bool), f["env_id"][:][off]
    valid = np.maximum(ln - cfg.eval.goal_offset_steps, 0)
    expected = {}
    for e, o, v in zip(env, orig, valid):
        if o and v:
            expected[int(e)] = expected.get(int(e), 0) + int(v)
    expected = {e: min(cfg.eval.per_env, v) for e, v in expected.items()}

    got_env = np.asarray(row_data["env_id"]).reshape(-1)
    envs, counts = np.unique(got_env, return_counts=True)
    assert dict(zip(envs.tolist(), counts.tolist())) == expected
    with h5py.File(STABLEWM_HOME / "datasets" / f"{cfg.train.dataset_name}.h5") as f:
        train_envs = set(f["env_id"][:][f["ep_offset"][:]].tolist())
    assert set(expected) <= train_envs  # val demos come from training envs only
