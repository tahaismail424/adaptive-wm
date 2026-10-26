import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by pusht-scenarios exports)
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # train.py, jepa.py, module.py live at the repo root

STABLEWM_HOME = Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser()
VARIED_PUSHT = STABLEWM_HOME / "datasets" / "varied_pusht"


def write_h5(path, ep_lens, state_value, parent_start=0, img=8):
    """Tiny file in the LeWM/swm HDF5 layout: T+1 rows per episode, NaN last action.

    `state_value` offsets every numeric column, so tests can tell which file
    (train or val) a sample came from after normalization.
    """
    ep_lens = np.asarray(ep_lens, dtype=np.int32)
    n = int(ep_lens.sum())
    rng = np.random.default_rng(int(state_value) + 100)
    action = rng.normal(state_value, 1.0, (n, 2)).astype(np.float32)
    action[np.cumsum(ep_lens) - 1] = np.nan
    cols = {
        "pixels": rng.integers(0, 255, (n, img, img, 3), dtype=np.uint8),
        "action": action,
        "state": rng.normal(state_value, 2.0, (n, 7)),
        "proprio": rng.normal(state_value, 3.0, (n, 4)),
        "episode_idx": np.repeat(np.arange(len(ep_lens)), ep_lens),
        "step_idx": np.concatenate([np.arange(k) for k in ep_lens]),
        "parent_idx": np.repeat(parent_start + np.arange(len(ep_lens)), ep_lens),
        "ep_len": ep_lens,
        "ep_offset": np.concatenate([[0], np.cumsum(ep_lens)[:-1]]).astype(np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in cols.items():
            f.create_dataset(k, data=v)
    return cols


@pytest.fixture
def synthetic_home(tmp_path, monkeypatch):
    """A dataset cache with syn/train.h5 (values ~ +10) and syn/val.h5 (~ -10)."""
    monkeypatch.setenv("LOCAL_DATASET_DIR", str(tmp_path))  # train.py's cache_dir
    data = tmp_path / "datasets" / "syn"
    train = write_h5(data / "train.h5", [30, 25, 40, 22], state_value=10.0)
    val = write_h5(data / "val.h5", [24, 30], state_value=-10.0, parent_start=100)
    return {"root": tmp_path, "train": train, "val": val}
