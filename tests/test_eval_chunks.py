"""Per-env eval sampling, video plans, and chunked dataset evaluation."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import stable_worldmodel as swm

from conftest import STABLEWM_HOME, VARIED_PUSHT, write_h5
from eval import evaluate_in_chunks, make_chunks, sample_eval_starts, video_plan

EP_LENS = [30, 40, 28, 35, 50, 45, 20, 60]
ENV_OF_EPISODE = [7, 7, 8, 8, 9, 9, 9, 10]  # env 10 has a single episode


@pytest.fixture
def ds(tmp_path):
    path = tmp_path / "datasets" / "syn" / "x_test_eval.h5"
    write_h5(path, EP_LENS, state_value=0.0)
    with h5py.File(path, "a") as f:
        f.create_dataset("env_id", data=np.repeat(ENV_OF_EPISODE, EP_LENS))
    return swm.data.HDF5Dataset("syn/x_test_eval", keys_to_cache=["state"], cache_dir=tmp_path)


def env_of(episodes):
    return np.asarray(ENV_OF_EPISODE)[np.asarray(episodes, dtype=int)]


# ---------------------------------------------------------------- sampling


def test_per_env_gives_every_env_the_same_count(ds):
    rows, episodes, starts = sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=0, per_env=4)
    envs, counts = np.unique(env_of(episodes), return_counts=True)
    assert envs.tolist() == [7, 8, 9, 10] and (counts == 4).all()
    assert np.all(np.diff(rows) > 0) and len(set(rows.tolist())) == len(rows)
    for ep, start in zip(episodes, starts):
        assert start + 25 < EP_LENS[int(ep)]
    assert 6 not in episodes.tolist()  # 20 steps: never a valid start


def test_per_env_spreads_over_episodes(ds):
    _, episodes, _ = sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=0, per_env=2)
    for env in (7, 8, 9):  # 2 valid episodes each -> one start from each
        eps = episodes[env_of(episodes) == env]
        assert len(set(eps.tolist())) == 2
    assert set(episodes[env_of(episodes) == 10].tolist()) == {7}  # only episode available


def test_per_env_is_deterministic(ds):
    a = sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=3, per_env=3)
    b = sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=3, per_env=3)
    c = sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=4, per_env=3)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_per_env_too_few_starts(ds):
    # env 8: episodes of 28 and 35 steps -> 3 + 10 = 13 valid starts (others have more)
    sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=0, per_env=13)
    with pytest.raises(ValueError, match="env 8"):
        sample_eval_starts(ds, num_eval=0, goal_offset=25, seed=0, per_env=14)


def test_per_env_needs_env_id(synthetic_home):
    plain = swm.data.HDF5Dataset("syn/val", cache_dir=synthetic_home["root"])
    with pytest.raises(ValueError, match="env_id"):
        sample_eval_starts(plain, num_eval=0, goal_offset=5, seed=0, per_env=1)


def test_uniform_mode_unchanged(ds):
    rows, _, _ = sample_eval_starts(ds, num_eval=10, goal_offset=25, seed=0)
    assert len(rows) == 10


# ---------------------------------------------------------------- videos / chunks


def test_video_plan_modes():
    env_ids = np.array([7, 7, 8, 9, 9, 9])
    assert video_plan("full", env_ids).all() and video_plan(True, env_ids).all()
    assert not video_plan("none", env_ids).any() and not video_plan(False, env_ids).any()
    np.testing.assert_array_equal(video_plan("sample", env_ids), [1, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(video_plan("sample", env_ids, 2), [1, 1, 1, 1, 1, 0])
    with pytest.raises(ValueError):
        video_plan("some", env_ids)


def test_chunks_cover_everything_once():
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    chunks = make_chunks(mask, chunk_size=2)
    positions = np.concatenate([pos for pos, _ in chunks])
    assert sorted(positions.tolist()) == list(range(8))
    assert all(len(pos) <= 2 for pos, _ in chunks)
    for pos, with_video in chunks:
        assert (mask[pos] == with_video).all()  # never mixed


class FakeWorld:
    """Stands in for swm.World: records calls, succeeds on even episode ids."""

    def __init__(self, n, log):
        self.num_envs, self.log, self.policy = n, log, None

    def set_policy(self, policy):
        self.policy = policy
        policy.attached_to.append(self.num_envs)

    def evaluate(self, dataset, episodes_idx, start_steps, video, **kwargs):
        assert len(episodes_idx) == self.num_envs
        assert self.policy._next_init is None  # warm start cleared before every chunk
        self.policy._next_init = "stale"
        self.log.append((list(episodes_idx), video, kwargs))
        return {"episode_successes": np.array([e % 2 == 0 for e in episodes_idx])}


class FakePolicy:
    def __init__(self):
        self.attached_to, self._next_init = [], None


def test_evaluate_in_chunks(tmp_path):
    episodes = np.array([10, 11, 12, 13, 14, 15, 16])
    starts = np.arange(7)
    mask = np.array([1, 0, 0, 1, 0, 0, 0], dtype=bool)
    chunks = make_chunks(mask, chunk_size=3)  # video [0, 3]; then [1, 2, 4], [5, 6]
    log, policy, made = [], FakePolicy(), []

    def make_world(n):
        made.append(n)
        w = FakeWorld(n, log)
        w.set_policy(policy)
        return w

    successes, videos = evaluate_in_chunks(
        make_world, None, episodes, starts, chunks, video_dir=tmp_path, goal_offset=25, eval_budget=50,
    )
    np.testing.assert_array_equal(successes, episodes % 2 == 0)  # back in original order
    assert made == [2, 3]  # one World per chunk size, reused for the second size-2 chunk
    assert [len(e) for e, _, _ in log] == [2, 3, 2]
    assert log[0][1] == Path(tmp_path, "chunk_000") and log[1][1] is None and log[2][1] is None
    assert all(kw == {"goal_offset": 25, "eval_budget": 50} for _, _, kw in log)
    assert videos == [(0, str(tmp_path / "chunk_000" / "env_0.mp4")), (3, str(tmp_path / "chunk_000" / "env_1.mp4"))]


# ---------------------------------------------------------------- integration


@pytest.mark.skipif(
    not (VARIED_PUSHT / "pusht_v0_only_shape_I_test_eval.h5").exists(), reason="shape_I export missing"
)
def test_chunked_eval_on_real_data(tmp_path):
    """Real Worlds (pts/PushTBounded-v0, random policy): per-env sampling plus
    uneven chunks, with videos for the sampled chunk only."""
    from pusht_scenarios.env import variations

    data = swm.data.HDF5Dataset(
        "varied_pusht/pusht_v0_only_shape_I_test_eval", keys_to_cache=["state"], cache_dir=STABLEWM_HOME,
    )
    rows, episodes, starts = sample_eval_starts(data, num_eval=0, goal_offset=25, seed=0, per_env=1)
    env_ids = np.asarray(data.get_row_data(rows)["env_id"]).reshape(-1)
    assert len(rows) == 27 and len(set(env_ids.tolist())) == 27
    keep = np.arange(4)  # 4 envs is enough here
    rows, episodes, starts, env_ids = rows[keep], episodes[keep], starts[keep], env_ids[keep]

    mask = np.array([True, False, False, False])
    chunks = make_chunks(mask, chunk_size=2)  # [0] with video, [1, 2], [3]
    worlds = []

    def make_world(n):
        w = swm.World(
            env_name="pusht_scenarios.env:pts/PushTBounded-v0", num_envs=n,
            max_episode_steps=10, image_shape=(224, 224),
        )
        w.set_policy(swm.policy.RandomPolicy())
        worlds.append(w)
        return w

    successes, videos = evaluate_in_chunks(
        make_world, data, episodes, starts, chunks, video_dir=tmp_path, goal_offset=25, eval_budget=2,
    )
    assert successes.shape == (4,)
    assert [Path(v).name for _, v in videos] == ["env_0.mp4"] and Path(videos[0][1]).exists()
    assert sorted(w.num_envs for w in worlds) == [1, 2]  # chunks of size 1 share a World
    for w in worlds:
        for env in w.envs.envs:
            assert int(variations.flatten(env)["block.shape"]) == 5  # held-out 'I' restored
