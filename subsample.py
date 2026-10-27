"""Episode-level dataset restriction for training (pusht-scenarios exports).

`subsample_by_parent` keeps a fraction of the recorded demos ("parents") per
environment and every episode derived from them (the recording itself plus its
augmented copies), e.g. half the stock + half the slippery demos of the mixed
friction export: a data-matched, domain-randomization-style baseline.

Sampling rule (shared with pusht-scenarios, so both sides agree): within each
env_id, sort that env's parent demo ids by sha256(f"{seed}:{parent_id}") and
keep the first floor(fraction * n + 0.5). That is an exact count per env,
deterministic for a seed, different per seed, and nested across fractions (the
25% subset is inside the 50% subset). Parent ids are the stable demo_id strings
from <file>.episodes.jsonl (not parent_idx, which is per-export).

`compact_clip_index` stores swm's clip index as one int64 array instead of a
Python list of (episode, start) tuples. DataLoader workers only read it, but
reading Python objects touches their refcounts, which copies the pages into
every worker (copy-on-write); with millions of clips and ~44 workers that was
tens of GB of host RAM. Same clips, same order.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


def clip_index(lengths, span):
    """(episode, start) for every clip of `span` steps, as an (N, 2) int64 array,
    in swm's order: episodes in order, starts ascending, short episodes skipped."""
    lengths = np.asarray(lengths, dtype=np.int64)
    counts = np.maximum(lengths - span + 1, 0)
    episodes = np.repeat(np.arange(len(lengths), dtype=np.int64), counts)
    first = np.cumsum(counts) - counts  # position of each episode's first clip
    starts = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(first, counts)
    return np.stack([episodes, starts], axis=1)


def compact_clip_index(dataset):
    """Replace the dataset's list-of-tuples clip index with an int64 array."""
    if not isinstance(dataset.clip_indices, np.ndarray):
        dataset.clip_indices = clip_index(dataset.lengths, dataset.span)
    return dataset


class EpisodeSubsetHDF5Dataset(HDF5Dataset):
    """HDF5Dataset restricted to some episodes: clips and column reads cover only
    those episodes. `offsets` still point into the full file, so loading is
    unchanged; get_col_data (used to fit the normalizers) returns only the kept
    rows. Created by `restrict_episodes`, not directly."""

    def get_col_data(self, col):
        return super().get_col_data(col)[self._kept_rows]


def restrict_episodes(dataset, keep):
    """Keep only episodes where `keep` (bool, one per episode) is True."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(dataset.lengths),):
        raise ValueError(f"keep mask has shape {keep.shape}, expected ({len(dataset.lengths)},)")
    if not isinstance(dataset, HDF5Dataset):
        raise TypeError(f"restrict_episodes needs an HDF5Dataset, got {type(dataset).__name__}")
    lengths = np.asarray(dataset.lengths)[keep]
    offsets = np.asarray(dataset.offsets)[keep]
    dataset.__class__ = EpisodeSubsetHDF5Dataset
    dataset.lengths, dataset.offsets = lengths, offsets
    dataset._kept_rows = np.concatenate(
        [np.arange(o, o + n, dtype=np.int64) for o, n in zip(offsets, lengths)]
    ) if len(lengths) else np.zeros(0, dtype=np.int64)
    dataset.clip_indices = clip_index(lengths, dataset.span)
    return dataset


def episodes_jsonl(dataset):
    """<name>.episodes.jsonl next to a pusht-scenarios <name>.h5 export."""
    path = Path(dataset.h5_path)
    return path.with_name(path.stem + ".episodes.jsonl")


def read_parents(dataset):
    """(parent_id, env_id) for every episode, in episode order."""
    path = episodes_jsonl(dataset)
    if not path.exists():
        raise FileNotFoundError(f"subsampling needs {path} (written by pusht-scenarios' export)")
    n = len(dataset.lengths)
    parents, envs = [None] * n, np.full(n, -1, dtype=np.int64)
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            i = int(rec["episode_idx"])
            parents[i], envs[i] = str(rec["parent_id"]), int(rec["env_id"])
    missing = [i for i, p in enumerate(parents) if p is None]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} of {n} episodes (e.g. {missing[:3]})")
    return parents, envs


def _rank_key(seed, parent_id):
    return hashlib.sha256(f"{seed}:{parent_id}".encode()).hexdigest()


def choose_parents(parents_by_env, fraction, seed):
    """{env_id: sorted parent ids} -> {env_id: kept parent ids} (see module doc)."""
    kept = {}
    for env, parents in parents_by_env.items():
        f = fraction.get(env, fraction.get(str(env))) if isinstance(fraction, dict) else fraction
        if f is None:
            raise ValueError(f"subsample.fraction has no entry for env_id {env}")
        if not 0.0 <= float(f) <= 1.0:
            raise ValueError(f"subsample fraction for env {env} must be in [0, 1], got {f}")
        ranked = sorted(parents, key=lambda p: _rank_key(seed, p))
        kept[env] = sorted(ranked[: int(np.floor(float(f) * len(ranked) + 0.5))])
    return kept


def subsample_by_parent(dataset, fraction, seed):
    """Restrict `dataset` (a pusht-scenarios export) to a per-env fraction of its
    recorded demos plus all their augmented copies. Returns a summary dict
    (counts per env and the kept parent ids) for logging."""
    parents, envs = read_parents(dataset)
    by_env = {}
    for p, e in zip(parents, envs):
        by_env.setdefault(int(e), set()).add(p)
    kept = choose_parents({e: sorted(ps) for e, ps in sorted(by_env.items())}, fraction, seed)
    kept_set = {p for ps in kept.values() for p in ps}
    keep = np.array([p in kept_set for p in parents], dtype=bool)
    lengths = np.asarray(dataset.lengths)

    summary = {"fraction": fraction, "seed": seed, "file": str(dataset.h5_path), "envs": {}}
    for e in sorted(by_env):
        in_env = envs == e
        summary["envs"][e] = {
            "parents_total": len(by_env[e]),
            "parents_kept": len(kept[e]),
            "episodes_kept": int((keep & in_env).sum()),
            "episodes_total": int(in_env.sum()),
            "rows_kept": int(lengths[keep & in_env].sum()),
        }
    restrict_episodes(dataset, keep)
    summary["clips"] = int(len(dataset.clip_indices))
    summary["kept_parent_ids"] = {e: ps for e, ps in kept.items()}
    return summary


def format_summary(summary):
    lines = [f"subsample fraction={summary['fraction']} seed={summary['seed']}: {summary['clips']} clips"]
    for e, c in summary["envs"].items():
        lines.append(
            f"  env {e}: {c['parents_kept']}/{c['parents_total']} demos, "
            f"{c['episodes_kept']}/{c['episodes_total']} episodes, {c['rows_kept']} rows"
        )
    return "\n".join(lines)
