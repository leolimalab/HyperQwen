"""Make a text-only copy of an MLX Qwen3.8-27B (qwen3_5) checkpoint for mlx_vlm.

The 4-bit MLX conversion LM Studio ships carries the vision tower (879 MiB),
and on a 24 GB Mac that is the difference between fitting beside a drafter and
going into swap. mlx_vlm has a decoder-only `qwen3_5_text` model type but no
switch to skip the tower of a `qwen3_5` checkpoint, and its loader is strict,
so the tower has to leave the files. Tensors are copied by byte range from the
safetensors headers, so this never holds more than one tensor in memory and
never touches MLX. Keys keep their `language_model.` prefix, which is also what
tells mlx_vlm not to shift the norm weights a second time.

  python3 mac/mlx_text_only.py ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit \\
      ~/models/mlx/Qwen3.8-27B-MLX-4bit-text
"""

import json
import os
import shutil
import struct
import sys

# No image/video processor configs: mlx_vlm loads the checkpoint through
# AutoProcessor, which falls back to the plain tokenizer only when nothing names
# a vision processor (tokenizer_config.json's processor_class is dropped below).
COPY = ("tokenizer.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "special_tokens_map.json")
CHUNK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    for shard in shards:
        hdr, base = header(os.path.join(src, shard))
        keep = {k: v for k, v in hdr.items() if k != "__metadata__" and not k.startswith("vision_tower.")}
        out_hdr, offset = {}, 0
        for k in sorted(keep, key=lambda k: keep[k]["data_offsets"][0]):
            a, b = keep[k]["data_offsets"]
            out_hdr[k] = {"dtype": keep[k]["dtype"], "shape": keep[k]["shape"], "data_offsets": [offset, offset + b - a]}
            offset += b - a
        if "__metadata__" in hdr:
            out_hdr["__metadata__"] = hdr["__metadata__"]
        blob = json.dumps(out_hdr, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        with open(os.path.join(src, shard), "rb") as fi, open(os.path.join(dst, shard), "wb") as fo:
            fo.write(struct.pack("<Q", len(blob)) + blob)
            for k in sorted(keep, key=lambda k: keep[k]["data_offsets"][0]):
                a, b = keep[k]["data_offsets"]
                fi.seek(base + a)
                left = b - a
                while left:
                    buf = fi.read(min(CHUNK, left))
                    fo.write(buf)
                    left -= len(buf)
        for k in keep:
            index["weight_map"][k] = shard
        index["metadata"]["total_size"] += offset
        print(f"{shard}: kept {len(keep)} of {len(hdr) - ('__metadata__' in hdr)} tensors, {offset / 2**20:.0f} MiB", flush=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    text = dict(cfg["text_config"])
    text.update(model_type="qwen3_5_text", architectures=["Qwen3_5ForCausalLM"],
                tie_word_embeddings=cfg.get("tie_word_embeddings", False),
                eos_token_id=cfg.get("eos_token_id", text.get("eos_token_id")))
    for k in ("quantization", "quantization_config"):
        if k in cfg:
            text[k] = cfg[k]
    json.dump(text, open(os.path.join(dst, "config.json"), "w"), indent=2)
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    for f in COPY:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dst)
    tok = json.load(open(os.path.join(src, "tokenizer_config.json")))
    tok.pop("processor_class", None)
    json.dump(tok, open(os.path.join(dst, "tokenizer_config.json"), "w"), indent=2, ensure_ascii=False)
    print(f"wrote {dst}: {index['metadata']['total_size'] / 2**20:.0f} MiB of weights")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]))
