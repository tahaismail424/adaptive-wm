# adaptive-wm

A fork of LeWM (`lucas-maes/le-wm`, GitHub `tahaismail424/adaptive-wm`). It holds the **modeling, training and eval code** for adaptive / generalizable JEPA world models on Push-T under varied environment conditions.

- **Step 1 (current):** train and eval *plain* LeWM on the varied-condition Push-T datasets from the sibling package `pusht-scenarios`. This gives a baseline for how LeWM copes with held-out environments.
- **Step 2:** add adaptive variants while keeping plain LeWM working unchanged. The variants are a T2-z-style trainable per-environment latent z (arXiv 2610.22364) and/or an RMA-style adaptation module (arXiv 2107.04034).

Context source: `2026-10-25-180437-alright-friend-plz-read-this-chat-history-from-a.txt` in this repo, an exported Claude Code session from the `pusht-scenarios` repo. Its last answer (the final ~600 lines) is the handoff for this repo. This file distills it and adds what was verified here on 2026-10-25.

## Sibling repos (all in `..` = `/scratch/user/tahaismail424/klab/c2_wm_proj/`)

| Dir | What |
|---|---|
| `adaptive-wm/` | this repo (fork; currently identical to upstream LeWM at `8edfeb3`) |
| `le-wm/` | the user's working clone of upstream LeWM on the cluster. It has a working `.venv` (Python 3.10, torch 2.14, PyPI `stable-worldmodel` 0.1.1, stable-pretraining 0.1.7) and **Slurm scripts in `scripts/slurm/`** (`env.sh`, `train.sbatch`, `eval.sbatch`, `smoke.sbatch`). It also has small local edits: per-step train / per-epoch val logging, `log_every_n_steps`, and random-policy results dir. Port these when setting up this repo. |
| `pusht-scenarios/` | data package: varied-condition Push-T env (`pts/PushT-v0`, `pts/PushTBounded-v0`), gamepad teleop recorder, augmentation, suites, export to LeWM HDF5. Its `CLAUDE.md` is the authority on the env and the dataset format. Python ≥3.13, torch-free core. |
| `sensorimotor-world-model/` | SMWM (another world-model codebase, not used yet) |

**This repo's env (user, 2026-10-25):** Python 3.13 via uv. `stable-worldmodel` and `pusht-scenarios` are git deps. `uv.lock` resolved them to swm `21446f1` (the rev with the eval hook) and pusht-scenarios `79feeb6`. `pyproject.toml` has no `rev=`, so `uv lock --upgrade` would move them; pin `rev` to be safe. Checked 2026-10-25: the installed swm has the `reset_options_from_dataset` hook, and torch, hydra, einops, h5py and pusht_scenarios import. Missing: stable-pretraining, lightning, wandb and transformers (all come with the `stable-worldmodel[train]` extra), plus scikit-learn (`eval.py`, not in the extra) and `pytest`. Importing torch/swm from scratch is slow, more than 2 minutes on a cold cache.

**Data:** the three exports are in `$STABLEWM_HOME/datasets/varied_pusht/` (uploaded 2026-10-25).

## Research context

- **Hidden factors** (mass, friction) change dynamics. A latent z in the *predictor* should help with these.
- **Visible factors** (color, shape) change pixels. A predictor-side z can't fix an encoder seeing new pixels, so held-out color is the expected failure / negative control unless z also conditions the *encoder*. Shape changes both pixels and dynamics.
- Report hidden and visible factors separately. The single-level hold-out datasets exist for this.
- T2-z facts:
  - z is 32-D per environment, initialised U(0,1), clipped to [0,1]. It's injected as an extra token plus an added modulation vector.
  - Training: 300 model steps, then alternating 200 latent / 200 model steps. Latent LR 0.03, model LR 1e-5.
  - Test-time adaptation: freeze the model, start a fresh z, take 40 steps (LR 3.0/1.5/0.5/0.15, 10 steps each) on **1–2 trajectories** from the new environment.
  - Scale: 12–84 environments × 9–50 episodes.

## How LeWM is wired here

- **`train.py`** (Hydra, `config/train/lewm.yaml`; defaults `data: pusht`, `model: lewm`).
  - `swm.data.load_dataset(cfg.data.dataset.name, **rest)` loads the data. A z-score normalizer is fitted on the *training file* for each non-pixel column in `keys_to_load`.
  - `action_encoder.input_dim = frameskip × action_dim`.
  - `spt.data.random_split(train_split=0.9)` splits by window, not by episode, so noisy copies of a demo leak into val (see Gotchas).
  - `lejepa_forward` computes the loss: MSE(pred_emb, emb[:, num_preds:]) + 0.09 · SIGReg. NaN actions become 0.
  - One optimizer (`model_opt`, AdamW 5e-5) inside `spt.Module`.
  - `SaveCkptCallback` writes `save_pretrained(...)` every epoch.
- **`jepa.py` `JEPA`**:
  - `encode` runs a ViT-tiny (patch 14, 224 px) and takes the CLS token through `projector`. `action_encoder` (`module.Embedder`) embeds actions.
  - `predict` runs `ARPredictor`, then `pred_proj`.
  - Planning path: `get_cost` → `rollout` (autoregressive, `history_size=3`) → `criterion` (MSE of the last predicted embedding to the goal embedding).
- **`module.py` `ARPredictor`**: a causal transformer of `ConditionalBlock`s. The action embedding `c` drives **AdaLN-zero** (`adaLN_modulation(c)` gives shift/scale/gate for attention and MLP). This is the natural hook for z: `c = act_emb + MLP(z)`.
- **`eval.py`** (Hydra, `config/eval/pusht.yaml`):
  1. Builds `swm.World(env_name, num_envs=num_eval)` and loads `HDF5Dataset(eval.dataset_name)`.
  2. Fits sklearn `StandardScaler`s for `dataset.keys_to_cache` **on the eval dataset**.
  3. Samples `num_eval=50` start rows where a `goal_offset_steps=25` goal fits.
  4. Calls `world.evaluate(dataset=..., callables=...)` with CEM (300 samples, 30 iterations, top-k 30), horizon 5, action_block 5, budget 50.
  - Success is the env's `eval_state`: agent **and** block within 20 px / 20° of the goal-frame state.
- **Dataset name resolution:**
  - train: `load_dataset(name)` → `$STABLEWM_HOME/datasets/<name>` (full filename with extension; a relative path with subdirs works).
  - eval: `HDF5Dataset(name)` → `$STABLEWM_HOME/datasets/<name>.h5` (no extension).
  - On the cluster, `STABLEWM_HOME=/scratch/user/tahaismail424/.stable_wm`.

## The varied Push-T datasets (from pusht-scenarios)

**Location (planned):** `/scratch/user/tahaismail424/.stable_wm/datasets/varied_pusht/`. The user is exporting on their Mac and will copy the files over. As of 2026-10-25 the directory doesn't exist yet. Config names are then `varied_pusht/<file>`.

**v0 suite** (`pusht-scenarios/configs/suites/v0.yaml`): 4 factors × 3 levels, all 3⁴ = 81 combinations recorded by hand (6 demos each, every env with its own start layouts). The goal is fixed (center, 45°).

| factor | levels | variation keys |
|---|---|---|
| mass | light 0.5 / normal 1 / heavy 3 | `sim.block_mass` |
| friction | low (damping 1, floor 0.4) / high (damping 1, floor 2.0) / stop (swm default: no momentum) | `sim.damping`, `sim.floor_friction` |
| color | default / warm / night | `block.color`, `agent.color`, background… |
| shape | T / I / + | `block.shape` = 2 / 5 / 7 |

All envs share agent strength `sim.agent_max_force = 10000`. Mass only matters with a limited-strength agent. It's clearly visible under `stop` friction (a push moves a light / normal / heavy block 138 / 86 / 31 px) but nearly invisible under `low` (135 vs 131 px).

**The three first experiments** (suites already built in `pusht-scenarios/configs/generated/`):

| dataset name | held out | train envs | test envs | kind |
|---|---|---|---|---|
| `pusht_v0_only_shape_I` | shape = I | 54 | 27 | visible + dynamics |
| `pusht_v0_only_color_night` | color = night | 54 | 27 | visible only |
| `pusht_v0_combo_heavy_stop` (name per the export command used) | mass = heavy **and** friction = stop together | 72 | 9 | hidden, compositional |

**Files per dataset** (`pts-export`):
- `<name>_train.h5`: train-env episodes of train parent demos plus their augmented copies. Each recording gets 3 bases (original, random SE(2), mirror) × 25 DINO-WM noise copies.
- `<name>_val.h5`: train envs, held-out parent demos (a clean split by parent).
- `<name>_test_adapt.h5`: test envs, the 2 lowest-seed clean recordings per env (for T2-z test-time adaptation).
- `<name>_test_eval.h5`: test envs, the remaining ~4 clean successful recordings per env. These are the planning eval tasks.
- `<name>_<split>.episodes.jsonl` and `<name>.manifest.yaml`: env list with ids, levels, counts, columns.

**Format** (LeWM / swm layout):
- Columns:
  - `pixels`: 224×224×3 uint8, Blosc-LZ4 chunks of 100.
  - `action`: 2-D.
  - `state`: 7 values = agent xy, block xy, block angle, agent velocity xy.
  - `proprio`: 4 values = agent xy + agent velocity.
  - `episode_idx`, `step_idx`, `ep_len`, `ep_offset`.
- Each episode has T+1 rows; the last action is NaN.
- Extra columns (not loaded unless listed in `keys_to_load`):
  - `env_id`, `seed`, `demo_idx`, `parent_idx`, `is_original`, `clean`, `success`, `on_goal`, `noise_std`, `base_idx`;
  - `variation.*`: ground-truth factor values, e.g. `variation.sim.block_mass` (for RMA-style supervision).
- **`env_id`: train envs are 0..N_train−1 (contiguous, so it can index an `nn.Embedding`). Test envs are N_train and up.**
- Recorded v0 demos are short: about 49 steps on average. About 94% have ≥26 rows, enough for the 25-step goal offset.
- Our frames have **no red contact dots**; LeWM's original `pusht_expert_train.h5` frames do. Keep this in mind when fine-tuning from or comparing against LeWM's Push-T checkpoint.

## Status (2026-10-25)

- **Train/val split: done.** `train.build_datasets(cfg, rnd_gen)`: if `cfg.data.val_set` is set (outside the `dataset:` block), that file is the val set, using the same train-fitted transform. Otherwise it falls back to the 0.9/0.1 `random_split`. Data configs `config/train/data/pusht_{shapeI,night,heavy}.yaml` all set `val_set`. Verified on the exports: zero `parent_idx` overlap between train and val, val envs are a subset of train envs, and train `env_id`s are contiguous. Also ported from `le-wm`: per-step train / per-epoch val logging, `log_every_n_steps`, and `scripts/slurm/` (gitignored by the user; `env.sh` points at this repo).
- **Eval: done, tests pending.** `config/eval/pusht_{shapeI,night,heavy}.yaml` set `env_name: pusht_scenarios.env:pts/PushTBounded-v0`, `eval.dataset_name: varied_pusht/<name>_test_eval` and `train.dataset_name: varied_pusht/<name>_train` (scaler stats); they have no callables. `solver/cem.yaml` uses `stable_worldmodel.planning.CEMSolver`.
  - `eval.py` helpers: `fit_scalers`, `sample_eval_starts`, `load_manifest` (`<export>.manifest.yaml` next to the h5), and `per_env_results`.
  - **Per-env breakdown:** `env_id` is just a dataset column. It's read from the sampled start rows and grouped against `metrics["episode_successes"]`, which is aligned with `episodes_idx`. It never goes through the env or the model. The output gives by-env and by-factor-level counts and is written to the results txt plus `<filename>_per_env.json`.
  - Random-policy results go to `$STABLEWM_HOME/random/<env>/`.
  - **Per-env sampling + chunks + videos (user, 2026-10-25):**
    - `eval.per_env: k` samples k starts for every test `env_id`, round-robin over each env's demos. The default in the varied configs is 20, giving 540 starts for shape_I and night and 180 for heavy. `num_eval` only applies when `per_env` is null.
    - `eval.chunk_size` (100) is the number of envs per `World` / `world.evaluate` call. There is one World per chunk size, because swm needs `num_envs == len(episodes)`. `solver.batch_size: ${eval.chunk_size}` makes CEM plan a whole chunk in one GPU batch. CEM's `batch_size` was 1 in LeWM, i.e. sequential planning.
    - Before every chunk the policy is re-attached (`set_policy` rebuilds the action buffers) and `policy._next_init = None` (the CEM warm start, which `set_env` does not reset). Otherwise state leaks across chunks.
    - **In-distribution val evals (user, 2026-10-25):** `config/eval/pusht_{shapeI,night,heavy}_val.yaml` are identical to the test configs except:
      - `eval.dataset_name: <name>_val`;
      - `per_env: 10`;
      - `only_original: true`: start only from `is_original` rows. `clean` is not enough, because each recording has 3 noise-free copies;
      - `allow_fewer: true`: up to k starts per env; envs without valid starts are skipped.
      - shape_I val: 32 recordings in 26 of the 54 train envs, mostly 1 per env, giving about 230 starts.
      - Compare val vs test to separate "can't plan on this data" from "doesn't generalize".
      - Videos go to `<results>/videos_<output filename stem>/`, so test and val evals of the same model don't overwrite each other. The random shapeI run from before this change wrote to `<results>/videos/`.
    - **Val loss caveat:** the val loader isn't shuffled (`shuffle=False`), and a `_val.h5` file is read in episode order. So `validate/sigreg_loss` (about 100 vs about 1 on train) and therefore `validate/loss` are meaningless. Use `validate/pred_loss`. The user decided not to fix this for now, since val isn't used for checkpoint selection.
    - `eval.video: full | none | sample`. `sample` means `videos_per_env` (1) per env, evaluated in their own chunks. Videos go to `<results>/videos/chunk_XXX/env_i.mp4`, and `videos/index.json` maps each video to its row, episode, start, env_id, name and success.
- Tests: `tests/test_data_split.py`, `tests/test_eval.py` and `tests/test_eval_chunks.py`, 34 tests in all. Synthetic h5 fixtures are in `tests/conftest.py`; real-data tests skip when the exports are missing. On 2026-10-25 all 34 passed, after adding `imageio[ffmpeg]` (swm's video writer needs it, and it isn't in the `[train]` extra). **A full run takes about 19 min on the login node, about 16 of them imports.** The test calls themselves take about 2 min. Run a single pytest process, or use a compute node.
  - Test gotchas: `spt.data.random_split` returns stable_pretraining's own `Subset`, not torch's (check `.indices`). swm normalizes action as (frameskip·num_steps, 2) rows *before* reshaping it to (num_steps, frameskip·2). `gym.spec()` doesn't accept the `module:` prefix; `gym.make` does.
- First full run: `lewm_pusht_shapeI`, Slurm job 3219913 (A100), 50 epochs × 5804 steps at about 4.2 it/s, so about 25 min/epoch and about 21 h total.

## What must change to train/eval plain LeWM on these (original plan; mostly done, see Status)

**Training: config only.** Add one data config per dataset, e.g. `config/train/data/pusht_v0_shape_I.yaml`:

```yaml
dataset:
  num_steps: ${eval:'${num_preds} + ${history_size}'}
  frameskip: 5
  name: varied_pusht/pusht_v0_only_shape_I_train.h5
  keys_to_load: [pixels, action, proprio, state]
  keys_to_cache: [action, proprio, state]
```

Then run `python train.py data=pusht_v0_shape_I output_model_name=... subdir=...`. Optional: load the clean `_val.h5` as the val set instead of `random_split` (a small `train.py` change).

**Eval: four changes.**
1. **swm version.** `pts` envs restore each episode's physics, colors and shape through `PushT.reset_options_from_dataset(init_row, goal_row)`. Only **swm at git rev `21446f1ede6d5284e981bd7b47f432b994e6d812`** (what pusht-scenarios pins) calls this hook.
   - **PyPI `stable-worldmodel` 0.1.1 (in `le-wm/.venv`) does not.** Its dataset eval does `reset(seed)` followed by the `_set_state` / `_set_goal_state` callables, so held-out envs would **silently run as default Push-T**. Both versions report "0.1.1"; verified by reading both on 2026-10-25.
   - At that rev, `CEMSolver` moved to `stable_worldmodel.planning.solver` (there is no `stable_worldmodel.solver`), so `config/eval/solver/cem.yaml` `_target_` must become `stable_worldmodel.planning.CEMSolver`.
   - `load_dataset`, `HDF5Dataset(name, cache_dir, keys_to_cache)`, `wm.utils.load_pretrained` / `save_pretrained`, `PlanConfig` and `WorldModelPolicy` all still exist at that rev.
   - A planned (not yet run) smoke test: train and eval under the new rev.
2. **Env id.** Set `world.env_name: pusht_scenarios.env:pts/PushTBounded-v0`. The `module:` prefix makes gymnasium import pusht-scenarios, which registers the env. The bounded variant stops CEM from driving the agent off the board. With the hook present, the `callables` list is ignored (it can stay or go).
   - This requires **pusht-scenarios installed in this env**. Resolved 2026-10-25: the env moved to Python 3.13 and pusht-scenarios is a git dep.
3. **Normalisation stats.** Set `eval.dataset_name: varied_pusht/<name>_test_eval`, but fit the StandardScalers on the **train** file. Currently `stats_dataset = dataset`, with a commented-out `cfg.dataset.stats` hint. Add a key like `dataset.stats: varied_pusht/<name>_train` and use it. Otherwise eval normalises with test-env statistics, a silent mismatch with training.
4. **Per-environment breakdown.** Starts are sampled uniformly over rows of the test file, so a single success rate mixes all test envs. Group `episode_successes` by `env_id` (and its factor levels from the manifest). One eval call per test env also works, and is what T2-z adaptation needs anyway.

Everything else stays: 50 evals, budget 50, goal offset 25, CEM horizon 5 × action_block 5. The combo dataset has only 9 × ~4 = ~36 test-eval episodes, so 50 starts reuse episodes; fine, but note it.

## Planned design for adaptive variants (step 2; not started)

- Add new classes next to the existing ones instead of editing them: e.g. `AdaptiveJEPA(JEPA)` plus a context-aware predictor, selected by a Hydra group (`model=lewm` stays identical; `model=t2z` adds a context). Config sketch:
  ```yaml
  context: {enabled: true, dim: 32, num_envs: 54, inject: [modulation], target: predictor, init: uniform01}
  ```
  - z table: `nn.Embedding(num_envs, dim)` indexed by `env_id`, which must be added to `keys_to_load`; it's constant within a window.
  - Modulation injection: `c = act_emb + MLP(z)`.
  - Token injection: prepend a token. This needs care with the causal mask and `pos_embedding`.
  - Later targets: `encoder` (FiLM on patch embeddings, for visible factors) or `both`.
- **Two-stage training:** two param groups, the model (T2-z LR 1e-5; LeWM uses 5e-5) and z (LR 0.03, no weight decay). Schedule: 300 model steps, then alternate 200 z / 200 model, freezing the other group and clipping z to [0,1]. `spt.Module` takes named optimizers. It's unverified whether it can alternate on a schedule; otherwise use Lightning manual optimisation in a custom module.
- **Test-time adaptation in eval:** for each test env, start a fresh z, freeze the model, and take 40 steps of the JEPA prediction loss on that env's `_test_adapt.h5` episodes (filter by `env_id`). Then plan with z fixed. Plain LeWM uses the same per-env loop without adaptation, so numbers compare directly. swm only injects dataset columns it already tracks into the planner infos, so `env_id` won't reach `get_cost` unless the adapted z is set on the model per eval call.
- RMA-style variant: supervise an encoder of history → factors using the `variation.*` columns.
- Checkpoints must round-trip through `save_pretrained` / `load_pretrained` (`_object.ckpt`), like `JEPA`.

## Environment / cluster

- Grace cluster (TAMU HPRC). Use `le-wm/scripts/slurm/env.sh` as the template: `module load uv/0.11.1 WebProxy/0000`, `STABLEWM_HOME=/scratch/user/tahaismail424/.stable_wm`, `SPT_CACHE_DIR=$STABLEWM_HOME/spt_runs`, `MUJOCO_GL=egl`. Training: 1 A100, 24 CPUs; eval: T4.
- The scratch file-count quota is 250k, so be careful with venvs and caches.
- **Never launch large cluster jobs.** Write the sbatch / command and hand it to the user.
- To read the pinned swm source: `git init && git fetch --depth 1 https://github.com/galilai-group/stable-worldmodel 21446f1ede6d5284e981bd7b47f432b994e6d812 && git checkout FETCH_HEAD` in a temp dir.

## Gotchas
- **Cluster / login node:** `uv` is not on PATH. Use `module load uv/0.11.1` (or `/sw/eb/sw/uv/0.11.1/bin/uv`). Lustre `/scratch` is slow (93% full). A cold `import torch` takes about 2 min, and `import lightning` (torchmetrics/transformers) more than 10 min cold. **Never run several torch-importing processes at once on the 1-CPU login node:** they starve each other and look hung (kernel state `I`, about 0% CPU).
- Reading any column of the exported h5 files with h5py needs `import hdf5plugin`, because even small columns are Blosc-compressed.
- Don't name a scratch script `inspect.py` (it shadows the stdlib module and numpy's import breaks).

- `train.py`'s `random_split` is by window, so augmented copies of a demo land in both train and val. This inflates val metrics only; the clean `_val.h5` fixes it.
- `eval.py` writes results next to the policy checkpoint (`$STABLEWM_HOME/<policy dir>/`), appending to `cfg.output.filename`.
- `eval.py` sets `world.max_episode_steps = 2 × eval_budget`.
- Don't use swm's `swm/PushT-v1` for varied data: it has none of the `sim.*` knobs.

## Working style

- The user works **deliberately and step by step**: explain options, agree, then implement. Don't race ahead into later phases (e.g. don't build T2-z while we're still on the baseline).
- When asked to "help me understand", explain; don't modify code unless asked.
- Keep plain LeWM behaviour intact; add, don't replace.
- Keep this file the single source of truth. Update it when decisions change; `AGENTS.md` only points here.
