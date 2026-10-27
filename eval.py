import os

os.environ["MUJOCO_GL"] = "egl"

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import yaml

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def fit_scalers(stats_dataset, keys):
    """StandardScalers for the policy's inputs, fitted on `stats_dataset` (the
    training file when given, so eval normalizes exactly like training)."""
    process = {}
    for col in keys:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_starts(dataset, num_eval, goal_offset, seed, per_env=None,
                       only_original=False, allow_fewer=False):
    """Pick eval start rows of `dataset` whose episode still has `goal_offset`
    steps after them. Returns (rows, episode ids, start steps), sorted by row.

    per_env=None: `num_eval` rows uniformly over all valid rows (LeWM's protocol).
    per_env=k: k rows for every `env_id` in the dataset, spread round-robin over
    each env's episodes (random episode order, random valid start within each),
    so every env and factor level gets the same weight. `num_eval` is ignored.

    only_original: start only from recorded demos (`is_original` column), not
    their augmented copies -- e.g. for planning evals on a `_val.h5` file.
    allow_fewer: with per_env, take up to k starts per env (envs with fewer
    valid starts give all they have) instead of raising.
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    if only_original:
        if "is_original" not in dataset.column_names:
            raise ValueError("only_original needs an is_original column (pusht-scenarios exports)")
        valid_mask &= np.asarray(dataset.get_col_data("is_original")).reshape(-1).astype(bool)
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")
    g = np.random.default_rng(seed)

    if per_env is None:
        if len(valid_indices) - 1 < num_eval:
            raise ValueError("Not enough episodes with sufficient length for evaluation.")
        random_episode_indices = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
        rows = valid_indices[random_episode_indices]
    else:
        rows = _per_env_rows(dataset, valid_indices, col_name, per_env, g, allow_fewer)

    # sort increasingly to avoid issues with HDF5Dataset indexing
    rows = np.sort(rows)
    row_data = dataset.get_row_data(rows)
    return rows, row_data[col_name], row_data["step_idx"]


def _per_env_rows(dataset, valid_indices, col_name, per_env, g, allow_fewer=False):
    if "env_id" not in dataset.column_names:
        raise ValueError("per_env sampling needs an env_id column (pusht-scenarios exports)")
    env_of_row = np.asarray(dataset.get_col_data("env_id")).reshape(-1)[valid_indices]
    ep_of_row = np.asarray(dataset.get_col_data(col_name)).reshape(-1)[valid_indices]
    rows, short = [], []
    for env_id in np.unique(env_of_row):
        in_env = env_of_row == env_id
        n_env = min(per_env, int(in_env.sum()))
        if n_env < per_env:
            if not allow_fewer:
                raise ValueError(f"env {env_id} has {in_env.sum()} valid starts, fewer than per_env={per_env}")
            short.append((int(env_id), n_env))
        # each episode's valid rows, shuffled; then take one per episode in turn
        episodes = g.permutation(np.unique(ep_of_row[in_env]))
        pools = [list(g.permutation(valid_indices[in_env & (ep_of_row == ep)])) for ep in episodes]
        picked = []
        while len(picked) < n_env:
            for pool in pools:
                if pool and len(picked) < n_env:
                    picked.append(pool.pop())
        rows.extend(picked)
    if short:
        print(f"{len(short)} envs have fewer than {per_env} valid starts (env, n): {short}")
    return np.asarray(rows, dtype=np.int64)


def video_plan(mode, env_ids, per_env_videos=1):
    """Which eval episodes get a video. mode: 'full' (all), 'none', or 'sample'
    (the first `per_env_videos` episodes of every env). Also accepts True/False
    for full/none."""
    if mode is True or mode == "full":
        return np.ones(len(env_ids), dtype=bool)
    if mode is False or mode is None or mode == "none":
        return np.zeros(len(env_ids), dtype=bool)
    if mode != "sample":
        raise ValueError(f"eval.video must be full|none|sample, got {mode!r}")
    mask = np.zeros(len(env_ids), dtype=bool)
    for env_id in np.unique(env_ids):
        mask[np.flatnonzero(env_ids == env_id)[:per_env_videos]] = True
    return mask


def make_chunks(video_mask, chunk_size):
    """Split episode positions into chunks of at most `chunk_size`, video and
    non-video episodes in separate chunks. Returns [(positions, with_video)]."""
    chunks = []
    for with_video in (True, False):
        pos = np.flatnonzero(video_mask == with_video)
        for i in range(0, len(pos), chunk_size):
            chunks.append((pos[i : i + chunk_size], with_video))
    return chunks


def evaluate_in_chunks(make_world, dataset, episodes, starts, chunks, video_dir=None, **evaluate_kwargs):
    """Run swm's dataset eval chunk by chunk (one World per chunk size, since
    swm needs num_envs == episodes per call). Returns per-episode successes in
    the original order and, for chunks with video, a video index."""
    worlds = {}
    successes = np.zeros(len(episodes), dtype=bool)
    videos = []
    for k, (pos, with_video) in enumerate(chunks):
        n = len(pos)
        if n not in worlds:
            worlds[n] = make_world(n)
        chunk_video = Path(video_dir, f"chunk_{k:03d}") if (with_video and video_dir) else None
        # re-attach the policy: rebuilds its per-env action/history buffers, so no
        # queued actions or CEM warm starts leak from the previous chunk
        policy = getattr(worlds[n], "policy", None)
        if policy is not None:
            worlds[n].set_policy(policy)
            if hasattr(policy, "_next_init"):  # WorldModelPolicy's CEM warm start; set_env keeps it
                policy._next_init = None
        metrics = worlds[n].evaluate(
            dataset=dataset,
            episodes_idx=[int(e) for e in episodes[pos]],
            start_steps=[int(s) for s in starts[pos]],
            video=chunk_video,
            **evaluate_kwargs,
        )
        successes[pos] = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
        if chunk_video is not None:
            videos += [(int(p), str(chunk_video / f"env_{i}.mp4")) for i, p in enumerate(pos)]
        print(f"chunk {k + 1}/{len(chunks)}: {successes[pos].sum()}/{n} successes")
    return successes, videos


def load_manifest(dataset):
    """pusht-scenarios writes <export>.manifest.yaml next to <export>_<split>.h5."""
    h5_path = Path(dataset.h5_path)
    stem = re.sub(r"_(train|val|test_adapt|test_eval)$", "", h5_path.stem)
    path = h5_path.with_name(f"{stem}.manifest.yaml")
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def per_env_results(dataset, rows, successes, manifest=None):
    """Group per-episode successes by the eval rows' `env_id` column.

    env_id never goes through the env or the model: it is a dataset column, so
    each eval episode's env is looked up from the row it started at. With the
    pusht-scenarios manifest, also report names, factor levels and per-level rates.
    Returns None for datasets without an env_id column (e.g. LeWM's pusht).
    """
    if "env_id" not in dataset.column_names:
        return None
    env_ids = np.asarray(dataset.get_row_data(rows)["env_id"]).reshape(-1).astype(int)
    successes = np.asarray(successes, dtype=bool).reshape(-1)
    envs = {e["env_id"]: e for e in (manifest or {}).get("envs", [])}

    by_env = {}
    for env_id in np.unique(env_ids):
        hit = successes[env_ids == env_id]
        info = envs.get(int(env_id), {})
        by_env[int(env_id)] = {
            "name": info.get("name"),
            "levels": info.get("levels"),
            "episodes": int(hit.size),
            "successes": int(hit.sum()),
            "success_rate": 100.0 * float(hit.mean()),
        }

    by_level = defaultdict(lambda: [0, 0])  # (factor, level) -> [successes, episodes]
    for env_id, success in zip(env_ids, successes):
        for factor, level in (envs.get(int(env_id), {}).get("levels") or {}).items():
            by_level[(factor, level)][0] += int(success)
            by_level[(factor, level)][1] += 1
    by_factor = defaultdict(dict)
    for (factor, level), (hits, n) in sorted(by_level.items()):
        by_factor[factor][level] = {"episodes": n, "successes": hits, "success_rate": 100.0 * hits / n}

    return {"by_env": by_env, "by_factor": dict(by_factor)}


def stats_dataset_name(cfg):
    """Dataset whose statistics normalize the policy's inputs (action, proprio,
    state): `train.dataset_name` if set, else None (= the eval dataset itself,
    LeWM's default). `train.dataset_name: from_checkpoint` reads the training
    file from the checkpoint run's config.yaml (written by train.py next to the
    weights), so each checkpoint is normalized exactly like it was trained --
    essential when evaluating checkpoints on data from another condition."""
    name = OmegaConf.select(cfg, "train.dataset_name")
    if name != "from_checkpoint":
        return name
    if cfg.get("policy", "random") == "random":
        return None  # a random policy doesn't use the scalers
    config_path = _checkpoint_config_path(cfg)
    trained_on = OmegaConf.select(OmegaConf.load(config_path), "data.dataset.name")
    if not trained_on:
        raise ValueError(f"{config_path} has no data.dataset.name")
    trained_on = str(trained_on)
    return trained_on[: -len(".h5")] if trained_on.endswith(".h5") else trained_on  # HDF5Dataset adds .h5


def _checkpoint_config_path(cfg):
    policy = Path(str(cfg.policy))
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), policy.parent if policy.suffix else policy)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(
            f"train.dataset_name=from_checkpoint, but {config_path} does not exist; "
            f"set train.dataset_name to the checkpoint's training file explicitly"
        )
    return config_path


def checkpoint_subsample(cfg):
    """The checkpoint run's data.subsample block ({fraction, seed}) when stats
    come from_checkpoint and the run trained on a subsample, else None. eval
    then rebuilds the same subset (the rule is deterministic) before fitting the
    scalers, so they match training exactly."""
    if OmegaConf.select(cfg, "train.dataset_name") != "from_checkpoint":
        return None
    if cfg.get("policy", "random") == "random":
        return None
    sub = OmegaConf.select(OmegaConf.load(_checkpoint_config_path(cfg)), "data.subsample")
    return OmegaConf.to_container(sub, resolve=True) if sub else None


def build_policy(cfg, process, transform):
    """Random policy, or the checkpoint `cfg.policy` planned with `cfg.solver`
    (on the solver's device)."""
    if cfg.get("policy", "random") == "random":
        return swm.policy.RandomPolicy()
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to(cfg.solver.get("device", "cuda"))
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    # swm (pinned rev) solvers take `cost=` (a Costable); JEPA exposes get_cost itself
    solver = hydra.utils.instantiate(cfg.solver, cost=model)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


def apply_variation_overrides(world, overrides):
    """Force variation values (dotted keys, e.g. {'sim.block_mass': 3.0}) on every
    env of `world`, on top of whatever each eval episode restores from its
    dataset row. Needs envs with reset_options_from_dataset (pts/PushT*), which
    swm's dataset eval calls to build each env's reset options."""
    if not overrides:
        return world
    from pusht_scenarios.env.variations import cast_values

    for env in world.envs.envs:
        base = env.unwrapped
        if not hasattr(base, "reset_options_from_dataset"):
            raise ValueError(
                f"variation_overrides need an env with reset_options_from_dataset "
                f"(pusht_scenarios pts/PushT*), got {type(base).__name__}"
            )
        values = cast_values(base, dict(overrides))  # validates keys, casts dtypes
        restore = base.reset_options_from_dataset

        def with_overrides(init_row, goal_row, _restore=restore, _values=values):
            options = _restore(init_row, goal_row)
            options.setdefault("variation_values", {}).update(_values)
            return options

        base.reset_options_from_dataset = with_overrides
    return world


def sweep_points(cfg):
    """[(name, overrides)] to evaluate. Without eval.sweep: one unnamed point with
    eval.variation_overrides. With eval.sweep (a list of {name, overrides}):
    one point each, its overrides layered on top of eval.variation_overrides."""
    def plain(value):  # OmegaConf node -> python; missing/None -> empty
        if value is None:
            return None
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    base = dict(plain(cfg.eval.get("variation_overrides")) or {})
    sweep = plain(cfg.eval.get("sweep"))
    if not sweep:
        return [(None, base)]
    points = []
    for point in sweep:
        if "name" not in point:
            raise ValueError(f"every eval.sweep point needs a name: {point}")
        points.append((str(point["name"]), {**base, **(point.get("overrides") or {})}))
    names = [n for n, _ in points]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate eval.sweep names: {names}")
    return points


def format_sweep(summary):
    lines = [f"{'point':<22} {'success':>9} {'rate':>7}   overrides"]
    for r in summary:
        lines.append(
            f"{(r['name'] or '-'):<22} {r['successes']:>4}/{r['num_episodes']:<4} {r['success_rate']:6.1f}%   {r['overrides']}"
        )
    return "\n".join(lines)


def output_filename(filename, policy):
    """Results file name for one eval: the configured name plus the checkpoint,
    e.g. pusht_shapeI_val_results.txt + lewm_pusht_shapeI/weights_epoch_5.pt
    -> pusht_shapeI_val_results_weights_epoch_5.txt. Checkpoints of one run share
    a results dir, so this keeps their txt/json/videos apart. Random: unchanged."""
    if not policy or policy == "random":
        return filename
    name = Path(filename)
    tag = Path(str(policy).rstrip("/")).stem
    return f"{name.stem}_{tag}{name.suffix}"


def format_per_env(results):
    lines = ["env_id  success      name"]
    for env_id, r in results["by_env"].items():
        lines.append(f"{env_id:>6}  {r['successes']:>2}/{r['episodes']:<2} {r['success_rate']:5.1f}%  {r['name'] or ''}")
    for factor, levels in results["by_factor"].items():
        cells = ", ".join(f"{lvl} {r['successes']}/{r['episodes']}" for lvl, r in levels.items())
        lines.append(f"by {factor}: {cells}")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # world environments are created per chunk size below
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    # normalize with the training file's statistics if given (held-out test files
    # have different stats); otherwise with the eval file itself (LeWM's default)
    train_name = stats_dataset_name(cfg)
    print(f"normalization stats from: {train_name or cfg.eval.dataset_name}")
    stats_dataset = get_dataset(cfg, train_name) if train_name else dataset
    sub = checkpoint_subsample(cfg)
    if sub:
        from subsample import format_summary, subsample_by_parent

        print(format_summary(subsample_by_parent(stats_dataset, sub["fraction"], sub.get("seed", 0))))
    process = fit_scalers(stats_dataset, cfg.dataset.keys_to_cache)

    # -- run evaluation
    policy = build_policy(cfg, process, transform)

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(swm.data.utils.get_cache_dir(), "random", cfg.world.env_name.replace("/", "_").replace(":", "_"))
    )

    # sample the episodes and the starting indices
    eval_rows, eval_episodes, eval_start_idx = sample_eval_starts(
        dataset, cfg.eval.num_eval, cfg.eval.goal_offset_steps, cfg.seed,
        per_env=cfg.eval.get("per_env"),
        only_original=cfg.eval.get("only_original", False),
        allow_fewer=cfg.eval.get("allow_fewer", False),
    )
    n_eval = len(eval_rows)
    print(f"{n_eval} eval episodes")

    env_ids = (
        np.asarray(dataset.get_row_data(eval_rows)["env_id"]).reshape(-1)
        if "env_id" in dataset.column_names else np.zeros(n_eval, dtype=int)
    )
    video_mask = video_plan(cfg.eval.get("video", "full"), env_ids, cfg.eval.get("videos_per_env", 1))
    chunks = make_chunks(video_mask, cfg.eval.get("chunk_size") or n_eval)

    results_path.mkdir(parents=True, exist_ok=True)
    # one results/video name per eval (test / val / ...) and checkpoint
    out_name = output_filename(cfg.output.filename, cfg.policy)
    callables = cfg.eval.get("callables")
    callables = OmegaConf.to_container(callables, resolve=True) if callables else None
    manifest = load_manifest(dataset)
    manifest_envs = {e["env_id"]: e for e in (manifest or {}).get("envs", [])}

    points = sweep_points(cfg)
    summary = []
    for name, overrides in points:
        stem = Path(out_name).stem + (f"_{name}" if name else "")
        point_name = stem + Path(out_name).suffix
        video_dir = results_path / f"videos_{stem}"
        if name:
            print(f"==== sweep point {name}: {overrides}")

        def make_world(n, _overrides=overrides):
            world = swm.World(**{**cfg.world, "num_envs": n}, image_shape=(224, 224))
            apply_variation_overrides(world, _overrides)
            world.set_policy(policy)
            return world

        start_time = time.time()
        successes, videos = evaluate_in_chunks(
            make_world, dataset, eval_episodes, eval_start_idx, chunks,
            video_dir=video_dir,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            callables=callables,
        )
        end_time = time.time()

        metrics = {
            "success_rate": 100.0 * float(successes.mean()),
            "num_episodes": n_eval,
            "episode_successes": successes,
        }
        print(f"success_rate{f' [{name}]' if name else ''}: {metrics['success_rate']:.1f}% over {n_eval} episodes")
        summary.append({
            "name": name, "overrides": overrides, "success_rate": metrics["success_rate"],
            "successes": int(successes.sum()), "num_episodes": n_eval,
            "evaluation_time": end_time - start_time,
        })

        # per-environment breakdown (pusht-scenarios exports carry an env_id column)
        env_results = per_env_results(dataset, eval_rows, successes, manifest)
        if env_results is not None:
            print(format_per_env(env_results))

        if videos:
            index = [
                {
                    "video": v, "row": int(eval_rows[p]), "episode": int(eval_episodes[p]),
                    "start": int(eval_start_idx[p]), "env_id": int(env_ids[p]),
                    "success": bool(successes[p]),
                    "name": manifest_envs.get(int(env_ids[p]), {}).get("name"),
                    "overrides": overrides,
                }
                for p, v in videos
            ]
            video_dir.mkdir(parents=True, exist_ok=True)
            with (video_dir / "index.json").open("w") as f:
                json.dump(index, f, indent=2)

        point_path = results_path / point_name
        with point_path.open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            if overrides:
                f.write(f"variation_overrides: {overrides}\n")
            f.write(f"metrics: success_rate={metrics['success_rate']:.2f} num_episodes={n_eval}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
            if env_results is not None:
                f.write("==== PER ENV ====\n")
                f.write(format_per_env(env_results) + "\n")

        if env_results is not None:
            with point_path.with_name(point_path.stem + "_per_env.json").open("w") as f:
                json.dump({"metrics": {"success_rate": metrics["success_rate"], "num_episodes": n_eval}, **env_results}, f, indent=2)

    if len(points) > 1 or points[0][0]:
        table = format_sweep(summary)
        print(table)
        sweep_path = results_path / (Path(out_name).stem + "_sweep.txt")
        with sweep_path.open("a") as f:
            f.write(f"\n==== {cfg.policy} on {cfg.eval.dataset_name} ====\n{table}\n")
        with sweep_path.with_suffix(".json").open("w") as f:
            json.dump({"policy": cfg.policy, "dataset": cfg.eval.dataset_name, "points": summary}, f, indent=2)


if __name__ == "__main__":
    run()
