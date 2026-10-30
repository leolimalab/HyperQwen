#!/bin/bash
# Qwen3.8-27B on an Apple Silicon Mac with mlx_vlm (MLX), the alternative to
# mac/serve.sh (llama.cpp). mlx_lm cannot speculate on this hybrid model (its
# DeltaNet cache does not roll back a rejected draft); mlx_vlm >= 0.7 can, with
# the same two drafters the 3090 arms use:
#  - SPEC=mtp (default here): Qwen's MTP head as a standalone 4-bit MLX
#    checkpoint (0.24 GB; it reuses the target's embeddings), block size 3.
#  - SPEC=dflash: the DFlash2 block drafter, 4-bit MLX conversion of
#    z-lab/Qwen3.8-27B-DFlash2 (block size 8), 1 GB.
# There is no lookup (n-gram) drafter in mlx_vlm, so the copy-the-prompt speedup
# of LOOKUP=1 does not carry over here.
#
# The target is a text-only copy of LM Studio's 4-bit MLX conversion
# (mac/mlx_text_only.py, 14.1 GiB): with the 879 MiB vision tower and the
# DFlash2 drafter, the first request of the first boot here pushed the Mac 2.7 GB
# into swap before the watchdog stopped it. MLX grows the KV cache with the context instead of reserving it,
# so CTX is the budget the preflight reserves, not a hard cap — the swap
# watchdog in mac/common.sh is what stops a context that outgrows it.
#
#   bash mac/download.sh mlx-text mlx-mtp      # text-only target + MTP drafter
#   CHECK_ONLY=1 bash mac/serve_mlx.sh         # preflight only, loads nothing
#   bash mac/serve_mlx.sh                      # SPEC=mtp, :18020
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/mac/common.sh"
LMS_MODELS="${LMS_MODELS:-$HOME/.lmstudio/models}"
DRAFTERS="${DRAFTERS:-$HOME/models/mlx-drafters}"   # outside LM Studio's folder: these are not chat models

MODEL="${MODEL:-$HOME/models/mlx/Qwen3.8-27B-MLX-4bit-text}"
DFLASH_DRAFT="${DFLASH_DRAFT:-$DRAFTERS/Qwen3.8-27B-DFlash2-MLX-4bit}"
MTP_DRAFT="${MTP_DRAFT:-$DRAFTERS/Qwen3.8-27B-MTP-4bit}"

SPEC="${SPEC:-mtp}"                 # mtp | dflash | off
DRAFT_BLOCK="${DRAFT_BLOCK:-}"      # override the drafter's block size (dflash 8, mtp 3)
KV_BITS="${KV_BITS:-}"              # empty = fp16 KV; 8 = uniform int8
CTX="${CTX:-16384}"                 # tokens the preflight budgets for
PREFILL_STEP="${PREFILL_STEP:-512}"   # prompt tokens per step; mlx_vlm's 2048 quadruples the activation peak
PORT="${PORT:-18020}"
BIND="${BIND:-127.0.0.1}"           # mlx_vlm's own default is 0.0.0.0
WATCHDOG="${WATCHDOG:-1}"
SWAP_LIMIT_MB="${SWAP_LIMIT_MB:-256}"   # tighter than llama.cpp's: MLX materializes the weights on the first request
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$SPEC" in
  dflash) DRAFT="$DFLASH_DRAFT"; KIND=dflash ;;
  mtp)    DRAFT="$MTP_DRAFT";    KIND=mtp ;;
  off)    DRAFT="";              KIND="" ;;
  *) echo "SPEC must be dflash, mtp or off (got '$SPEC')" >&2; exit 2 ;;
esac
for d in "$MODEL" ${DRAFT:+"$DRAFT"}; do
  [ -f "$d/config.json" ] || { echo "missing $d — run: bash mac/download.sh mlx-text mlx-mtp (or mlx-dflash)" >&2; exit 1; }
done

# --- memory estimate --------------------------------------------------------
# KV on the 16 full-attention layers only: 4 heads x 256 dims, K and V.
MODEL_MB=$(dir_mib "$MODEL")
DRAFT_MB=$([ -n "$DRAFT" ] && dir_mib "$DRAFT" || echo 0)
KV_BYTES=$([ "$KV_BITS" = 8 ] && echo 1 || echo 2)
KV_MB=$(( 16 * 2 * 4 * 256 * KV_BYTES * CTX / 1048576 ))
STATE_MB=300                                         # GDN state + the speculative rollback copy
COMPUTE_MB=$(( 800 + PREFILL_STEP / 2 ))             # prefill activations
[ -n "$DRAFT" ] && COMPUTE_MB=$(( COMPUTE_MB + 300 ))
NEED_MB=$(( MODEL_MB + DRAFT_MB + KV_MB + STATE_MB + COMPUTE_MB ))
GPU_NEED_MB=$NEED_MB

echo "== preflight (mlx_vlm $(python3 -c 'import mlx_vlm; print(mlx_vlm.__version__)' 2>/dev/null))"
echo "   target  $(basename "$MODEL")  ${MODEL_MB} MiB"
[ -n "$DRAFT" ] && echo "   drafter $(basename "$DRAFT")  ${DRAFT_MB} MiB  ($KIND${DRAFT_BLOCK:+, block $DRAFT_BLOCK})"
echo "   KV      ${KV_BITS:-16}-bit x ${CTX} tokens  ${KV_MB} MiB (grows with the context)"
mac_preflight

# --- launch -----------------------------------------------------------------
source "$REPO/resolve_api_key.sh"
resolve_vllm_key
ARGS=(--model "$MODEL" --host "$BIND" --port "$PORT"
      --max-num-seqs 1 --prefill-step-size "$PREFILL_STEP")
[ -n "$DRAFT" ] && ARGS+=(--draft-model "$DRAFT" --draft-kind "$KIND")
[ -n "$DRAFT_BLOCK" ] && ARGS+=(--draft-block-size "$DRAFT_BLOCK")
[ -n "$KV_BITS" ] && ARGS+=(--kv-bits "$KV_BITS" --kv-quant-scheme uniform)
[ -n "${VLLM_API_KEY:-}" ] && ARGS+=(--api-key "$VLLM_API_KEY")
# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

echo "== mlx_vlm.server ${ARGS[*]}" | sed 's/--api-key [^ ]*/--api-key ***/'
# APC (mlx_vlm's prefix cache) stays off, its default: its block pool is sized
# for machines with room to spare.
mac_run_watched env APC_ENABLED=0 mlx_vlm.server "${ARGS[@]}"
