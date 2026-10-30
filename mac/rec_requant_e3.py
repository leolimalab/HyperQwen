"""Requantize the routed experts of Recursive's flint-int4 MLX checkpoint to 3 bits.

rec (Recursive's engine: resident Metal megakernel, expert and KV kernels) is
what reaches 125+ tok/s on Flint, but the official checkpoint is 22.6 GB and
the engine refuses Macs under 48 GiB. The fork thinkspread/rec@local/flint-e3-24gb
lifts the gate and accepts `routed_expert_bits` 3, which the upstream loader,
megakernel and mixed-bits kernels already handle; nobody publishes such a
checkpoint, so this makes one. The routed experts are 21.3 of the 22.6 GB, and
at 3 bits the checkpoint drops to ~17.8 GB, ~1.4 GB read per token.

Only tensors under `.switch_mlp.` change: INT4 group-64 is dequantized and
quantized again at 3 bits, group 64. Dense projections stay INT4 and the
embeddings INT8. The original export picked its scales by MSE
(`scale_rule: mse`); `--rule mse` (the default) does the same against the INT4
weights: it pulls each end of a group's range in by 0-40 % (5 x 5 steps) and
keeps, per group, the clip with the lowest squared error. On layer 22 that is
2.88 % relative error against 3.46 % for plain mx.quantize (`--rule minmax`);
a symmetric search reached 2.99 % and a 9 x 9 grid 2.85 % at three times the
cost. Weights are stored as U8 like the original; the engine views them
as uint32.

Each shard is read tensor by tensor and written to a `.partial` file that is
renamed when complete, so memory stays near one expert block. With
`--delete-src` each source shard is removed once its output is in place, which
keeps the peak disk use near the size of the source checkpoint.

Run it with the rec engine's venv (MLX 0.32):

  ~/.venvs/rec-e3/bin/python -I mac/rec_requant_e3.py ~/models/flint-int4 ~/models/flint-int4-e3 --delete-src
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import time

import mlx.core as mx
import numpy as np

COPY = ("chat_template.jinja", "generation_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "LICENSE")
GROUP = 64
SHRINKS = (1.0, 0.9, 0.8, 0.7, 0.6)   # each end of the range, from the group's midpoint
CLIPS = [(lo, hi) for lo in SHRINKS for hi in SHRINKS]
EXPERTS_PER_STEP = 32
READ_CHUNK = 64 << 20
NP_DTYPES = {"BF16": np.uint16, "U8": np.uint8, "F32": np.float32, "U32": np.uint32}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(f, base, info):
    a, b = info["data_offsets"]
    f.seek(base + a)
    raw = np.frombuffer(f.read(b - a), dtype=NP_DTYPES[info["dtype"]]).reshape(info["shape"])
    arr = mx.array(raw)
    if info["dtype"] == "BF16":
        arr = arr.view(mx.bfloat16)
    elif info["dtype"] == "U8":
        arr = arr.view(mx.uint32)
    return arr


def to_bytes(arr):
    if arr.dtype == mx.bfloat16:
        arr = arr.view(mx.uint16)
    elif arr.dtype == mx.uint32:
        arr = arr.view(mx.uint8)
    return np.array(arr).tobytes()


def requant_block(q4, s4, b4, rule):
    """INT4 g64 -> INT3 g64 for a slice of experts; returns (q3, s3, b3, err_minmax, err_best, energy)."""
    w = mx.dequantize(q4, s4, b4, group_size=GROUP, bits=4)          # bf16, the weights rec runs today
    shape = w.shape
    wg = w.astype(mx.float32).reshape(*shape[:-1], -1, GROUP)
    energy = (wg * wg).sum()
    lo, hi = wg.min(axis=-1, keepdims=True), wg.max(axis=-1, keepdims=True)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    best = None
    for lo_shrink, hi_shrink in CLIPS if rule == "mse" else [(1.0, 1.0)]:
        clipped = wg
        if (lo_shrink, hi_shrink) != (1.0, 1.0):
            clipped = mx.clip(wg, mid - half * lo_shrink, mid + half * hi_shrink)
        q, s, b = mx.quantize(clipped.reshape(shape).astype(mx.bfloat16), group_size=GROUP, bits=3)
        d = mx.dequantize(q, s, b, group_size=GROUP, bits=3).astype(mx.float32).reshape(wg.shape)
        err = ((d - wg) ** 2).sum(axis=-1)                           # one value per group
        qg = q.view(mx.uint8).reshape(*err.shape, GROUP * 3 // 8)    # 24 bytes per group
        if best is None:
            best = [err, qg, s, b]
            err_minmax = err.sum()
        else:
            better = err < best[0]
            best = [mx.where(better, err, best[0]),
                    mx.where(better[..., None], qg, best[1]),
                    mx.where(better, s, best[2]),
                    mx.where(better, b, best[3])]
        mx.eval(best, err_minmax)
    err, qg, s, b = best
    q3 = qg.reshape(*shape[:-1], -1).view(mx.uint32)
    return q3, s, b, err_minmax, err.sum(), energy


def convert_shard(src, dst, rule, stats):
    hdr, base = header(src)
    meta = hdr.pop("__metadata__", None)
    names = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
    experts = sorted({k.rsplit(".", 1)[0] for k in names if ".switch_mlp." in k})
    plain = [k for k in names if ".switch_mlp." not in k]

    # Header first: the 3-bit shapes follow from the 4-bit ones (in*4/8 -> in*3/8 bytes).
    out, order, offset = {}, [], 0
    def place(name, dtype, shape):
        nonlocal offset
        n = int(np.prod(shape)) * (2 if dtype == "BF16" else np.dtype(NP_DTYPES[dtype]).itemsize)
        out[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        order.append(name)
        offset += n
    for k in plain:
        place(k, hdr[k]["dtype"], hdr[k]["shape"])
    for p in experts:
        wq = hdr[p + ".weight"]
        if wq["dtype"] != "U8":
            raise ValueError(f"{p}.weight is {wq['dtype']}, expected U8")
        place(p + ".weight", "U8", wq["shape"][:-1] + [wq["shape"][-1] * 3 // 4])
        place(p + ".scales", "BF16", hdr[p + ".scales"]["shape"])
        place(p + ".biases", "BF16", hdr[p + ".biases"]["shape"])
    if meta is not None:
        out["__metadata__"] = meta
    blob = json.dumps(out, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    partial = dst + ".partial"
    with open(src, "rb") as fi, open(partial, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)) + blob)
        for k in plain:
            a, b = hdr[k]["data_offsets"]
            fi.seek(base + a)
            left = b - a
            while left:
                buf = fi.read(min(READ_CHUNK, left))
                fo.write(buf)
                left -= len(buf)
        for p in experts:
            q4 = read_tensor(fi, base, hdr[p + ".weight"])
            s4 = read_tensor(fi, base, hdr[p + ".scales"])
            b4 = read_tensor(fi, base, hdr[p + ".biases"])
            parts = [[], [], []]
            for e in range(0, q4.shape[0], EXPERTS_PER_STEP):
                sl = slice(e, e + EXPERTS_PER_STEP)
                q3, s3, b3, e_mm, e_best, energy = requant_block(q4[sl], s4[sl], b4[sl], rule)
                for i, t in enumerate((q3, s3, b3)):
                    parts[i].append(to_bytes(t))
                stats["minmax"] += e_mm.item()
                stats["best"] += e_best.item()
                stats["energy"] += energy.item()
            for i, suffix in enumerate((".weight", ".scales", ".biases")):
                data = b"".join(parts[i])
                a, b = out[p + suffix]["data_offsets"]
                if len(data) != b - a:
                    raise RuntimeError(f"{p}{suffix}: wrote {len(data)} bytes, header says {b - a}")
                fo.write(data)
            del q4, s4, b4, parts
            mx.clear_cache()
        fo.flush()
        os.fsync(fo.fileno())
    expected = 8 + len(blob) + offset
    if os.path.getsize(partial) != expected:
        raise RuntimeError(f"{partial}: {os.path.getsize(partial)} bytes, expected {expected}")
    os.replace(partial, dst)
    return len(experts)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(READ_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="flint-int4 directory (RecursiveSI/flint-int4)")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--rule", choices=("mse", "minmax"), default="mse")
    ap.add_argument("--delete-src", action="store_true", help="remove each source shard once converted")
    args = ap.parse_args()
    src, dst = os.path.expanduser(args.src), os.path.expanduser(args.dst)
    os.makedirs(dst, exist_ok=True)

    config = json.load(open(os.path.join(src, "config.json")))
    export = config["mlx_export"]
    if (export["bits"], export["routed_expert_bits"], export["embedding_bits"]) != (4, 4, 8):
        raise SystemExit(f"expected the INT4 export, got mlx_export={export}")

    stats = {"minmax": 0.0, "best": 0.0, "energy": 0.0}
    shards = sorted(f for f in os.listdir(src) if f.startswith("model-") and f.endswith(".safetensors"))
    for shard in shards:
        t0 = time.time()
        n = convert_shard(os.path.join(src, shard), os.path.join(dst, shard), args.rule, stats)
        print(f"{shard}: {n} expert projections, {os.path.getsize(os.path.join(dst, shard)) / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f} s", flush=True)
        if args.delete_src:
            os.remove(os.path.join(src, shard))

    for name in COPY:
        if os.path.exists(os.path.join(src, name)):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    export.update(routed_expert_bits=3, routed_expert_scale_rule=f"{args.rule}, requantized from RecursiveSI/flint-int4")
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    source = json.load(open(os.path.join(src, "flint-int4.json")))
    # The same set as the official manifest: configuration, tokenizer and shards.
    files = sorted(f for f in os.listdir(dst) if (f in COPY and f != "LICENSE") or f == "config.json"
                   or (f.startswith("model-") and f.endswith(".safetensors")))
    manifest = {
        "model_id": "flint-int4-e3",
        "weight_format": source["weight_format"],
        "kv_format": source["kv_format"],
        "max_context": source["max_context"],
        "checkpoint_files": {f: sha256(os.path.join(dst, f)) for f in files},
    }
    with open(os.path.join(dst, "flint-int4-e3.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    rel_mm = stats["minmax"] / stats["energy"]
    rel_best = stats["best"] / stats["energy"]
    print(f"expert weights, error relative to INT4: min-max {rel_mm:.5f}, {args.rule} {rel_best:.5f}")
    print(f"wrote {dst} and its manifest flint-int4-e3.json ({len(files)} files)")


if __name__ == "__main__":
    main()
