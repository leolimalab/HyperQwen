#!/bin/bash
# Qwen3.8-27B on oMLX (the Apple Silicon server app, /Applications/oMLX.app),
# for comparison with mac/serve.sh (llama.cpp) and mac/serve_mlx.sh (mlx_vlm).
#
# oMLX keeps its settings in ~/.omlx; this script never touches them. It runs
# `omlx serve` against its own base path (~/models/omlx-bench), regenerated on
# every launch, and its own model folder, where the text-only target from
# mac/mlx_text_only.py is linked in (no copy) with a `qwen3_5` config minus the
# vision block: oMLX serves that through its mlx-lm engine as a text-only quant,
# whereas the `qwen3_5_text` type mlx_vlm wants has no mlx-lm module.
#
#  - SPEC=off (default) or SPEC=dflash: oMLX's DFlash (dflash_mlx, which knows
#    DFlash2) with the 4-bit MLX drafter from mac/download.sh mlx-dflash. A
#    pre-quantized drafter skips oMLX's quantize-on-load, which is also the
#    step that installs its tuned verify kernels; quantizing the bf16 drafter
#    instead would hold 3.85 GB of bf16 weights for a moment on a 24 GB Mac.
#  - The per-model settings mirror the ones in ~/.omlx/model_settings.json for
#    Qwen3.8-27B-MLX-4bit: TurboQuant 4-bit KV, DFlash block 5, verify "dflash".
#  - oMLX's native MTP needs the MTP head inside the checkpoint, and its VLM MTP
#    path needs the full vision model; neither fits here, so there is no SPEC=mtp.
#
#   CHECK_ONLY=1 bash mac/serve_omlx.sh
#   LOAD_GRACE_S=120 SPEC=dflash bash mac/serve_omlx.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/mac/common.sh"

TEXT_MODEL="${TEXT_MODEL:-$HOME/models/mlx/Qwen3.8-27B-MLX-4bit-text}"
SRC_CONFIG="${SRC_CONFIG:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit/config.json}"
DFLASH_DRAFT="${DFLASH_DRAFT:-$HOME/models/mlx-drafters/Qwen3.8-27B-DFlash2-MLX-4bit}"
MODELS="${MODELS:-$HOME/models/omlx-models}"
BASE="${BASE:-$HOME/models/omlx-bench}"
MODEL_ID="Qwen3.8-27B-MLX-4bit-text"

SPEC="${SPEC:-off}"                    # off | dflash
DFLASH_BLOCK="${DFLASH_BLOCK:-5}"      # your oMLX setting for this model
DFLASH_VERIFY="${DFLASH_VERIFY:-dflash}"  # dflash | adaptive | ddtree
TURBOQUANT_BITS="${TURBOQUANT_BITS:-4}"   # 0 = fp16 KV
CTX="${CTX:-16384}"
SSD_CACHE="${SSD_CACHE:-10GB}"         # oMLX prefix cache on disk; 0 = off
PORT="${PORT:-18020}"
BIND="${BIND:-127.0.0.1}"
WATCHDOG="${WATCHDOG:-1}"
SWAP_LIMIT_MB="${SWAP_LIMIT_MB:-256}"
OMLX="${OMLX:-/opt/homebrew/bin/omlx}"

case "$SPEC" in
  off) DRAFT="" ;;
  dflash) DRAFT="$DFLASH_DRAFT" ;;
  *) echo "SPEC must be off or dflash (got '$SPEC')" >&2; exit 2 ;;
esac
for d in "$TEXT_MODEL" ${DRAFT:+"$DRAFT"}; do
  [ -f "$d/config.json" ] || { echo "missing $d — run: bash mac/download.sh mlx-text mlx-dflash" >&2; exit 1; }
done
[ -x "$OMLX" ] || { echo "oMLX CLI not found at $OMLX" >&2; exit 1; }

# --- the oMLX-side model folder: links to the text-only weights -------------
M="$MODELS/$MODEL_ID"
mkdir -p "$M"
for f in "$TEXT_MODEL"/*.safetensors "$TEXT_MODEL"/model.safetensors.index.json "$TEXT_MODEL"/tokenizer* \
         "$TEXT_MODEL"/vocab.json "$TEXT_MODEL"/chat_template.jinja "$TEXT_MODEL"/generation_config.json; do
  [ -e "$f" ] && ln -sf "$f" "$M/$(basename "$f")"
done
python3 - "$SRC_CONFIG" "$M/config.json" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
for k in ("vision_config", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
    cfg.pop(k, None)
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
EOF

# --- memory estimate --------------------------------------------------------
MODEL_MB=$(dir_mib "$TEXT_MODEL")
DRAFT_MB=$([ -n "$DRAFT" ] && dir_mib "$DRAFT" || echo 0)
KV_BITS=$([ "$TURBOQUANT_BITS" -gt 0 ] && echo "$TURBOQUANT_BITS" || echo 16)
KV_MB=$(( 16 * 2 * 4 * 256 * KV_BITS / 8 * CTX / 1048576 ))
STATE_MB=300
COMPUTE_MB=1300
[ -n "$DRAFT" ] && COMPUTE_MB=1600
NEED_MB=$(( MODEL_MB + DRAFT_MB + KV_MB + STATE_MB + COMPUTE_MB ))
GPU_NEED_MB=$NEED_MB

echo "== preflight (oMLX $("$OMLX" --version 2>/dev/null | head -1))"
echo "   target  $MODEL_ID  ${MODEL_MB} MiB"
[ -n "$DRAFT" ] && echo "   drafter $(basename "$DRAFT")  ${DRAFT_MB} MiB  (dflash, block $DFLASH_BLOCK, verify $DFLASH_VERIFY)"
echo "   KV      ${KV_BITS}-bit x ${CTX} tokens  ${KV_MB} MiB"
mac_preflight

# --- oMLX base path: settings regenerated from the knobs ---------------------
source "$REPO/resolve_api_key.sh"
resolve_vllm_key
mkdir -p "$BASE"
python3 - "$BASE" "$MODELS" "$MODEL_ID" <<EOF
import json, os, sys
base, models, mid = sys.argv[1:4]
key = os.environ.get("VLLM_API_KEY", "")
settings = {
  "version": "1.0",
  "server": {"host": "$BIND", "port": $PORT, "log_level": "info", "auto_start_on_launch": False},
  "model": {"model_dirs": [models], "model_dir": models, "model_fallback": False},
  "memory": {"prefill_memory_guard": True, "memory_guard_tier": "balanced"},
  "scheduler": {"max_concurrent_requests": 1},
  "cache": {"enabled": "$SSD_CACHE" != "0", "ssd_cache_max_size": "$SSD_CACHE" if "$SSD_CACHE" != "0" else "auto"},
  "auth": {"api_key": key, "skip_api_key_verification": not key},
  "huggingface": {"hf_cache_enabled": False},
  # Qwen's recommended sampling, as llama.cpp reads it from the GGUF
  "sampling": {"max_context_window": $CTX, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
}
json.dump(settings, open(os.path.join(base, "settings.json"), "w"), indent=2)
model = {"turboquant_kv_enabled": $TURBOQUANT_BITS > 0, "turboquant_kv_bits": $TURBOQUANT_BITS or 4,
         "turboquant_skip_last": True, "specprefill_enabled": False, "mtp_enabled": False,
         "vlm_mtp_enabled": False, "dflash_enabled": "$SPEC" == "dflash", "is_pinned": False}
if "$SPEC" == "dflash":
    model.update(dflash_draft_model="$DFLASH_DRAFT", dflash_draft_quant_enabled=False,
                 dflash_block_size=$DFLASH_BLOCK, dflash_verify_mode="$DFLASH_VERIFY",
                 dflash_in_memory_cache=False, dflash_ssd_cache=False)
json.dump({"version": "1.0", "models": {mid: model}}, open(os.path.join(base, "model_settings.json"), "w"), indent=2)
EOF

ARGS=(serve --base-path "$BASE" --model-dir "$MODELS" --host "$BIND" --port "$PORT"
      --max-concurrent-requests 1 --memory-guard balanced --no-hf-cache)
[ "$SSD_CACHE" = 0 ] && ARGS+=(--no-cache) || ARGS+=(--paged-ssd-cache-dir "$BASE/cache" --paged-ssd-cache-max-size "$SSD_CACHE")
[ -n "${VLLM_API_KEY:-}" ] && ARGS+=(--api-key "$VLLM_API_KEY")
echo "== omlx ${ARGS[*]}  (model id: $MODEL_ID)" | sed 's/--api-key [^ ]*/--api-key ***/'
mac_run_watched "$OMLX" "${ARGS[@]}"
