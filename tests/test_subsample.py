"""Parent-level subsampling of pusht-scenarios exports (subsample.py), its use in
train.build_datasets / eval, the compact clip index, and an end-to-end CPU
training smoke run on the real DR-50 config."""

import json
import pickle

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from conftest import ROOT, STABLEWM_HOME, write_h5
from subsample import (
    choose_parents, clip_index, compact_clip_index, restrict_episodes, subsample_by_parent,
)

SPAN = 20  # frameskip 5 x num_steps 4
MIXED = STABLEWM_HOME / "datasets" / "friction_pusht" / "mixed"


def make_export(root, name, families, seed=0):
    """A pusht-scenarios-like export: `families` = [(env_id, parent_id, [episode lengths])],
    the first episode of each family being the recording. Writes <name>.h5 and
    <name>.episodes.jsonl under root/datasets/syn/."""
    rng = np.random.default_rng(seed)
    lens, envs, parents, lines = [], [], [], []
    for env, parent, ep_lens in families:
        for j, n in enumerate(ep_lens):
            lines.append({"episode_idx": len(lens), "parent_id": parent, "env_id": env,
                          "demo_id": parent if j == 0 else f"{parent}-copy{j}"})
            lens.append(n)
            envs.append(env)
            parents.append(parent)
    path = root / "datasets" / "syn" / f"{name}.h5"
    write_h5(path, lens, state_value=float(rng.normal()))
    with h5py.File(path, "a") as f:
        f.create_dataset("env_id", data=np.repeat(envs, lens))
        f.create_dataset("is_original", data=np.repeat(
            [1 if line["demo_id"] == line["parent_id"] else 0 for line in lines], lens).astype(np.uint8))
    with path.with_name(f"{name}.episodes.jsonl").open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return lens, envs, parents


def families(seed=0):
    """env 0: 10 parents x 3 episodes; env 1: 7 parents x 2 episodes; some short."""
    rng = np.random.default_rng(seed)
    fam = [(0, f"a{i}", list(rng.integers(15, 60, size=3))) for i in range(10)]
    fam += [(1, f"b{i}", list(rng.integers(15, 60, size=2))) for i in range(7)]
    return fam


def load(root, name, **kw):
    return swm.data.load_dataset(f"syn/{name}.h5", transform=None, cache_dir=str(root), num_steps=4,
                                 frameskip=5, keys_to_load=["pixels", "action", "proprio", "state"],
                                 keys_to_cache=["action", "proprio", "state"], **kw)


@pytest.fixture
def export(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DATASET_DIR", str(tmp_path))
    lens, envs, parents = make_export(tmp_path, "mix_train", families())
    make_export(tmp_path, "mix_val", [(0, "v0", [40]), (1, "v1", [45])], seed=1)
    return {"root": tmp_path, "lens": np.array(lens), "envs": np.array(envs), "parents": parents}


# ---------------------------------------------------------------- clip index


@pytest.mark.parametrize("lengths", [[5, 20, 21, 50, 19, 33], [20], [3, 4], [100, 1, 100]])
def test_clip_index_matches_swm_order(lengths):
    expected = [(ep, s) for ep, n in enumerate(lengths) if n >= SPAN for s in range(n - SPAN + 1)]
    got = clip_index(lengths, SPAN)
    assert got.dtype == np.int64 and got.shape == (len(expected), 2)
    assert [tuple(x) for x in got.tolist()] == expected


def test_compact_clip_index_gives_identical_samples(export):
    plain = load(export["root"], "mix_train")
    compact = compact_clip_index(load(export["root"], "mix_train"))
    assert isinstance(compact.clip_indices, np.ndarray) and len(compact) == len(plain)
    for i in [0, 7, len(plain) // 2, len(plain) - 1]:
        a, b = plain[i], compact[i]
        for k in a:  # action ends in NaN (each episode's last action) by design
            torch.testing.assert_close(a[k], b[k], equal_nan=True)


# ---------------------------------------------------------------- the sampling rule


def test_exact_count_per_env_with_rounding():
    by_env = {0: [f"a{i}" for i in range(198)], 1: [f"b{i}" for i in range(189)]}
    kept = choose_parents(by_env, {0: 0.5, 1: 0.5}, seed=0)
    assert (len(kept[0]), len(kept[1])) == (99, 95)  # 94.5 rounds up, as in pusht-scenarios


def test_deterministic_seeded_and_nested():
    by_env = {0: [f"p{i}" for i in range(40)]}
    a, b = choose_parents(by_env, 0.5, 0), choose_parents(by_env, 0.5, 0)
    other = choose_parents(by_env, 0.5, 1)
    assert a == b and a != other
    quarter, half, full = (set(choose_parents(by_env, f, 3)[0]) for f in (0.25, 0.5, 1.0))
    assert quarter < half < full == set(by_env[0])


@pytest.mark.parametrize("fraction", [0.5, {0: 0.5, 1: 0.5}, {"0": 0.5, "1": 0.5}])
def test_fraction_forms(fraction):
    kept = choose_parents({0: ["x", "y"], 1: ["z", "w"]}, fraction, seed=0)
    assert len(kept[0]) == 1 and len(kept[1]) == 1


@pytest.mark.parametrize("fraction, match", [({0: 0.5}, "no entry for env_id 1"), (1.5, r"\[0, 1\]")])
def test_bad_fractions(fraction, match):
    with pytest.raises(ValueError, match=match):
        choose_parents({0: ["x"], 1: ["y"]}, fraction, seed=0)


# ---------------------------------------------------------------- subsetting a dataset


def test_subsample_keeps_whole_families(export):
    ds = load(export["root"], "mix_train")
    summary = subsample_by_parent(ds, {0: 0.5, 1: 0.5}, seed=0)
    kept = {p for ps in summary["kept_parent_ids"].values() for p in ps}
    assert summary["envs"][0]["parents_kept"] == 5 and summary["envs"][1]["parents_kept"] == 4

    keep = np.array([p in kept for p in export["parents"]])
    assert len(ds.lengths) == keep.sum()  # every episode of a kept parent, none of the others
    np.testing.assert_array_equal(ds.lengths, export["lens"][keep])
    for env in (0, 1):
        assert summary["envs"][env]["episodes_kept"] == int((keep & (export["envs"] == env)).sum())

    expected_clips = int(np.maximum(export["lens"][keep] - SPAN + 1, 0).sum())
    assert len(ds) == summary["clips"] == expected_clips


def test_subset_samples_are_the_full_files_rows(export):
    full = load(export["root"], "mix_train")
    sub = load(export["root"], "mix_train")
    subsample_by_parent(sub, 0.5, seed=0)
    # map each kept episode back to its index in the full file through its offset
    full_ep = {int(o): i for i, o in enumerate(np.asarray(full.offsets))}
    for i in [0, len(sub) // 3, len(sub) - 1]:
        ep, start = sub.clip_indices[i]
        a = sub[i]
        b = full._load_slice(full_ep[int(sub.offsets[ep])], int(start), int(start) + SPAN)
        torch.testing.assert_close(a["state"], b["state"])
        torch.testing.assert_close(a["pixels"], b["pixels"])
        assert start + SPAN <= sub.lengths[ep]  # clip stays inside its episode


def test_columns_cover_only_kept_rows(export):
    ds = load(export["root"], "mix_train")
    full_state = ds.get_col_data("state").copy()
    subsample_by_parent(ds, 0.5, seed=0)
    rows = np.concatenate([np.arange(o, o + n) for o, n in zip(ds.offsets, ds.lengths)])
    np.testing.assert_array_equal(ds.get_col_data("state"), full_state[rows])
    assert ds.get_dim("action") == 2


def test_subset_pickles_and_loads_in_workers(export):
    ds = load(export["root"], "mix_train")
    subsample_by_parent(ds, 0.5, seed=0)
    clone = pickle.loads(pickle.dumps(ds))  # spawn-safe
    assert len(clone) == len(ds)
    torch.testing.assert_close(clone[3]["state"], ds[3]["state"])
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, shuffle=True)
    batch = next(iter(loader))
    assert batch["pixels"].shape[:2] == (4, 4)


def test_restrict_episodes_validates(export):
    ds = load(export["root"], "mix_train")
    with pytest.raises(ValueError, match="shape"):
        restrict_episodes(ds, np.ones(3, dtype=bool))


def test_missing_jsonl_raises(export):
    (export["root"] / "datasets/syn/mix_train.episodes.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="episodes.jsonl"):
        subsample_by_parent(load(export["root"], "mix_train"), 0.5, seed=0)


# ---------------------------------------------------------------- train.build_datasets


def build_cfg(subsample=None):
    data = {
        "dataset": {"name": "syn/mix_train.h5", "num_steps": 4, "frameskip": 5,
                    "keys_to_load": ["pixels", "action", "proprio", "state"],
                    "keys_to_cache": ["action", "proprio", "state"]},
        "val_set": "syn/mix_val.h5",
    }
    if subsample:
        data["subsample"] = subsample
    return OmegaConf.create({"seed": 0, "train_split": 0.9, "img_size": 8, "data": data,
                             "model": {"action_encoder": {"input_dim": "???"}}})


def test_build_datasets_without_subsample_is_unchanged(export):
    from train import build_datasets

    train_set, val_set = build_datasets(build_cfg(), torch.Generator())
    plain = load(export["root"], "mix_train")
    assert train_set.subsample_summary is None
    assert [tuple(x) for x in train_set.clip_indices.tolist()] == plain.clip_indices  # same clips, same order
    assert len(val_set) == len(load(export["root"], "mix_val"))


def test_build_datasets_with_subsample(export):
    from train import build_datasets

    cfg = build_cfg({"fraction": {0: 0.5, 1: 0.5}, "seed": 0})
    train_set, val_set = build_datasets(cfg, torch.Generator())
    summary = train_set.subsample_summary
    assert summary["envs"][0]["parents_kept"] == 5 and len(train_set) == summary["clips"]
    assert len(val_set) == len(load(export["root"], "mix_val"))  # val untouched
    # normalizers fit on the kept rows only
    kept_state = torch.from_numpy(train_set.get_col_data("state"))
    full_state = torch.from_numpy(load(export["root"], "mix_train").get_col_data("state"))
    raw = load(export["root"], "mix_train")
    subsample_by_parent(raw, {0: 0.5, 1: 0.5}, seed=0)
    i = 5
    expected = ((raw[i]["state"] - kept_state.mean(0)) / kept_state.std(0)).float()
    torch.testing.assert_close(train_set[i]["state"], expected)
    assert not torch.allclose(kept_state.mean(0), full_state.mean(0))


# ---------------------------------------------------------------- eval side


def test_eval_rebuilds_the_checkpoint_subsample(tmp_path, monkeypatch):
    from eval import checkpoint_subsample, stats_dataset_name

    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    run = tmp_path / "checkpoints" / "dr"
    run.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"data": {"dataset": {"name": "x/mix_train.h5"},
                                              "subsample": {"fraction": {0: 0.5, 1: 0.5}, "seed": 2}}}),
                   run / "config.yaml")
    cfg = OmegaConf.create({"policy": "dr/weights_epoch_10.pt", "train": {"dataset_name": "from_checkpoint"}})
    assert stats_dataset_name(cfg) == "x/mix_train"
    assert checkpoint_subsample(cfg) == {"fraction": {0: 0.5, 1: 0.5}, "seed": 2}
    cfg.train.dataset_name = "x/explicit_train"  # explicit stats file: no subsample implied
    assert checkpoint_subsample(cfg) is None


# ---------------------------------------------------------------- real data


requires_mixed = pytest.mark.skipif(
    not (MIXED / "pusht_v05_friction_mixed_train.episodes.jsonl").exists(), reason="mixed export missing"
)


def compose_train(overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        return compose("lewm", overrides=list(overrides))


def test_dr50_config():
    cfg = compose_train(["data=pusht_friction_dr50"])
    mixed = compose_train(["data=pusht_friction_mixed"])
    assert cfg.data.dataset == mixed.data.dataset and cfg.data.val_set == mixed.data.val_set
    assert OmegaConf.to_container(cfg.data.subsample) == {"fraction": {0: 0.5, 1: 0.5}, "seed": 0}


@requires_mixed
def test_dr50_on_the_real_mixed_export():
    from train import build_datasets

    cfg = compose_train(["data=pusht_friction_dr50"])
    train_set, val_set = build_datasets(cfg, torch.Generator().manual_seed(0))
    s = train_set.subsample_summary
    assert (s["envs"][0]["parents_kept"], s["envs"][0]["parents_total"]) == (99, 198)
    assert (s["envs"][1]["parents_kept"], s["envs"][1]["parents_total"]) == (95, 189)
    # whole families: about half the episodes of each floor
    for env in (0, 1):
        frac = s["envs"][env]["episodes_kept"] / s["envs"][env]["episodes_total"]
        assert 0.4 < frac < 0.6
    # same step budget as the single-floor models (stock ~19.2k / slippery ~16.6k steps per epoch)
    steps = s["clips"] // 128
    assert 14_000 < steps < 22_000, steps
    full_val = swm.data.load_dataset(cfg.data.val_set, transform=None, num_steps=4, frameskip=5,
                                     keys_to_load=["state"], keys_to_cache=["state"])
    assert len(val_set) == len(full_val)
    sample = train_set[len(train_set) // 2]
    assert sample["pixels"].shape == (4, 3, 224, 224) and torch.isfinite(sample["state"]).all()


@requires_mixed
def test_train_smoke_run_on_cpu(tmp_path, monkeypatch):
    """train.run on the DR-50 config for 2 train + 1 val batches on CPU: the real
    loader, normalizers, model, optimizer and checkpointing, plus subsample.json.
    Then eval's from_checkpoint stats rebuild exactly the same subset."""
    import train as train_mod
    from eval import checkpoint_subsample, get_dataset, stats_dataset_name
    from subsample import subsample_by_parent as sub_fn

    (tmp_path / "datasets").symlink_to(STABLEWM_HOME / "datasets")
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path / "spt"))
    monkeypatch.delenv("LOCAL_DATASET_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = compose_train([
        "data=pusht_friction_dr50", "output_model_name=smoke_dr50", "subdir=smoke_dr50",
        "trainer.accelerator=cpu", "trainer.devices=1", "trainer.precision=32", "trainer.max_epochs=1",
        "+trainer.limit_train_batches=2", "+trainer.limit_val_batches=1",
        "loader.batch_size=4", "num_workers=2", "loader.persistent_workers=false",
        "loader.prefetch_factor=2", "loader.pin_memory=false", "wandb.enabled=false",
    ])
    train_mod.run(cfg)

    run_dir = tmp_path / "checkpoints" / "smoke_dr50"
    assert (run_dir / "config.yaml").exists() and (run_dir / "weights_epoch_1.pt").exists()
    saved = json.loads((run_dir / "subsample.json").read_text())
    assert saved["envs"]["0"]["parents_kept"] == 99 and saved["envs"]["1"]["parents_kept"] == 95

    eval_cfg = OmegaConf.create({"policy": "smoke_dr50/weights_epoch_1.pt", "cache_dir": None,
                                 "train": {"dataset_name": "from_checkpoint"},
                                 "dataset": {"keys_to_cache": ["action", "proprio", "state"]}})
    name = stats_dataset_name(eval_cfg)
    assert name == "friction_pusht/mixed/pusht_v05_friction_mixed_train"
    stats = get_dataset(eval_cfg, name)
    rebuilt = sub_fn(stats, **checkpoint_subsample(eval_cfg))
    assert {str(k): v for k, v in rebuilt["kept_parent_ids"].items()} == saved["kept_parent_ids"]
