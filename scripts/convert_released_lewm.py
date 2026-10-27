"""Convert a released LeWM checkpoint (HF `quentinll/lewm-*`, saved with
transformers 4.x ViT parameter names) into this repo's checkpoint format, so
eval.py loads it as a plain `jepa.JEPA` like any checkpoint trained here.

The released weights are architecturally identical to ours; only the ViT
encoder's per-layer parameter names differ (transformers 5.x renamed them):

    encoder.encoder.layer.i.attention.attention.{query,key,value} -> encoder.layers.i.attention.{q,k,v}_proj
    encoder.encoder.layer.i.attention.output.dense                -> encoder.layers.i.attention.o_proj
    encoder.encoder.layer.i.intermediate.dense                    -> encoder.layers.i.mlp.fc1
    encoder.encoder.layer.i.output.dense                          -> encoder.layers.i.mlp.fc2
    encoder.encoder.layer.i.layernorm_{before,after}              -> encoder.layers.i.layernorm_{before,after}

Usage (paths relative to $STABLEWM_HOME/checkpoints/):
    python scripts/convert_released_lewm.py \
        --src models--quentinll--lewm-pusht/weights.pt \
        --reference lewm_pusht_shapeI/weights_epoch_30.pt \
        --out lewm_pusht_released

Then eval with POLICY=lewm_pusht_released/weights.pt. The output's config.json is
copied from the reference run; key names and shapes must match it exactly.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

RENAMES = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.(layernorm_before|layernorm_after)\.", r"encoder.layers.\1.\2."),
]


def rename_key(key: str) -> str:
    for pattern, repl in RENAMES:
        new, n = re.subn(pattern, repl, key)
        if n:
            return new
    return key


def convert(state_dict: dict, reference: dict) -> dict:
    """Rename keys and require an exact key/shape match with `reference`."""
    out = {rename_key(k): v for k, v in state_dict.items()}
    if len(out) != len(state_dict):
        raise ValueError("renaming produced duplicate keys")
    missing, extra = set(reference) - set(out), set(out) - set(reference)
    if missing or extra:
        raise ValueError(
            f"key mismatch after renaming: {len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(extra)} unexpected (e.g. {sorted(extra)[:3]})"
        )
    bad = [k for k in out if tuple(out[k].shape) != tuple(reference[k].shape)]
    if bad:
        raise ValueError(f"shape mismatch for {bad[:5]}")
    return out


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="released weights.pt (relative to checkpoints/ or absolute)")
    p.add_argument("--reference", required=True, help="a weights_*.pt from a run trained here, next to its config.json")
    p.add_argument("--out", required=True, help="output run dir name under checkpoints/")
    args = p.parse_args()

    root = Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser() / "checkpoints"
    src, ref = root / args.src, root / args.reference
    out_dir = root / args.out
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; pick another --out")

    converted = convert(torch.load(src, map_location="cpu"), torch.load(ref, map_location="cpu"))
    out_dir.mkdir(parents=True)
    torch.save(converted, out_dir / "weights.pt")
    shutil.copy(ref.parent / "config.json", out_dir / "config.json")
    print(f"wrote {out_dir}/weights.pt ({len(converted)} tensors) + config.json from {ref.parent.name}")


if __name__ == "__main__":
    main()
