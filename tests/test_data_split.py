"""Train/val splitting in train.build_datasets, plus sanity checks on the
pusht-scenarios exports in $STABLEWM_HOME/datasets/varied_pusht/."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
import yaml
from omegaconf import OmegaConf

from conftest import ROOT, STABLEWM_HOME, VARIED_PUSHT
from train import build_datasets

DATA_CONFIGS = sorted((ROOT / "config/train/data").glob("*.yaml"))
VARIED_CONFIGS = [p for p in DATA_CONFIGS if "varied_pusht/" in p.read_text()]
KEYS = ["pixels", "action", "proprio", "state"]


def make_cfg(val_set=None):
    data = {
        "dataset": {
            "name": "syn/train.h5",
            "num_steps": 4,
            "frameskip": 5,
            "keys_to_load": KEYS,
            "keys_to_cache": ["action", "proprio", "state"],
        }
    }
    if val_set:
        data["val_set"] = val_set
    return OmegaConf.create({
        "seed": 0,
        "train_split": 0.9,
        "img_size": 8,
        "data": data,
        "model": {"action_encoder": {"input_dim": "???"}},
    })


def raw_dataset(name, cache_dir):
    return swm.data.load_dataset(
        name, transform=None, cache_dir=cache_dir, num_steps=4, frameskip=5,
        keys_to_load=KEYS, keys_to_cache=["action", "proprio", "state"],
    )


def is_split(ds):
    """random_split returns a stable_pretraining Subset (not torch's): it has .indices."""
    return hasattr(ds, "indices")


def zscore_stats(col):
    """What utils.get_column_normalizer computes: NaN rows dropped, torch std."""
    data = torch.from_numpy(np.asarray(col))
    data = data[~torch.isnan(data).any(dim=1)]
    return data.mean(0), data.std(0)


# ---------------------------------------------------------------- synthetic


def test_val_file_is_the_val_set(synthetic_home):
    cfg = make_cfg(val_set="syn/val.h5")
    train_set, val_set = build_datasets(cfg, torch.Generator().manual_seed(0))

    root = synthetic_home["root"]
    assert len(train_set) == len(raw_dataset("syn/train.h5", root))  # whole file, no split
    assert len(val_set) == len(raw_dataset("syn/val.h5", root))
    assert not is_split(train_set)


def test_val_is_normalized_with_train_stats(synthetic_home):
    cfg = make_cfg(val_set="syn/val.h5")
    _, val_set = build_datasets(cfg, torch.Generator().manual_seed(0))
    raw_val = raw_dataset("syn/val.h5", synthetic_home["root"])

    for col in ["action", "proprio", "state"]:
        mean, std = zscore_stats(synthetic_home["train"][col])
        for i in [0, len(val_set) - 1]:
            raw = raw_val[i][col]
            # the normalizer runs before swm reshapes action to (num_steps, frameskip * 2)
            expected = ((raw.reshape(-1, mean.numel()) - mean) / std).float().reshape(raw.shape)
            got = val_set[i][col]
            torch.testing.assert_close(got, expected, equal_nan=True)
    # val data sits around -10 and train around +10: with train stats val is far off-center
    assert val_set[0]["state"].mean() < -3


def test_val_and_train_share_transform(synthetic_home):
    train_set, val_set = build_datasets(make_cfg(val_set="syn/val.h5"), torch.Generator())
    assert val_set.transform is train_set.transform


def test_action_encoder_input_dim(synthetic_home):
    cfg = make_cfg(val_set="syn/val.h5")
    build_datasets(cfg, torch.Generator())
    assert cfg.model.action_encoder.input_dim == 5 * 2  # frameskip * action dim


def test_no_val_set_falls_back_to_random_split(synthetic_home):
    cfg = make_cfg()
    train_set, val_set = build_datasets(cfg, torch.Generator().manual_seed(0))
    n = len(raw_dataset("syn/train.h5", synthetic_home["root"]))
    assert is_split(train_set) and is_split(val_set)
    assert len(train_set) + len(val_set) == n
    assert set(train_set.indices).isdisjoint(val_set.indices)
    assert abs(len(val_set) - 0.1 * n) <= 1


# ---------------------------------------------------------------- configs


@pytest.mark.parametrize("path", VARIED_CONFIGS, ids=lambda p: p.stem)
def test_varied_pusht_configs_use_exported_val_split(path):
    """Every varied_pusht data config must point val_set at the matching _val.h5,
    otherwise training silently falls back to the leaky random window split."""
    cfg = yaml.safe_load(path.read_text())
    train_name = cfg["dataset"]["name"]
    assert train_name.endswith("_train.h5")
    assert cfg.get("val_set") == train_name.replace("_train.h5", "_val.h5")
    assert "val_set" not in cfg["dataset"]  # would be passed to load_dataset as a kwarg


# ---------------------------------------------------------------- real exports

requires_data = pytest.mark.skipif(
    not VARIED_PUSHT.exists(), reason=f"{VARIED_PUSHT} not present"
)


def episode_values(path, col):
    with h5py.File(path, "r") as f:
        return f[col][:][f["ep_offset"][:]]


@requires_data
@pytest.mark.parametrize("path", VARIED_CONFIGS, ids=lambda p: p.stem)
def test_export_has_no_parent_leakage(path):
    cfg = yaml.safe_load(path.read_text())
    datasets = STABLEWM_HOME / "datasets"
    train, val = datasets / cfg["dataset"]["name"], datasets / cfg["val_set"]
    if not (train.exists() and val.exists()):
        pytest.skip("export files missing")

    # demo/parent indices are numbered across the whole export, so they compare across files
    assert set(episode_values(train, "parent_idx")).isdisjoint(episode_values(val, "parent_idx"))
    train_envs = set(episode_values(train, "env_id"))
    assert set(episode_values(val, "env_id")) <= train_envs
    assert train_envs == set(range(len(train_envs)))  # contiguous 0..N-1 (latent-table index)


@requires_data
def test_shape_I_config_builds():
    """The real Hydra config composes and loads (pixels are read lazily)."""
    from hydra import compose, initialize_config_dir

    shape_I = VARIED_PUSHT / "pusht_v0_only_shape_I_train.h5"
    if not shape_I.exists():
        pytest.skip("shape_I export missing")
    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        cfg = compose("lewm", overrides=["data=pusht_shapeI"])
    train_set, val_set = build_datasets(cfg, torch.Generator().manual_seed(cfg.seed))

    assert not is_split(train_set)
    assert 0 < len(val_set) < 0.2 * len(train_set)
    assert cfg.model.action_encoder.input_dim == 10
    sample = val_set[0]
    assert sample["pixels"].shape == (4, 3, 224, 224)
    for col in ["action", "proprio", "state"]:
        assert torch.isfinite(sample[col]).all() or col == "action"
