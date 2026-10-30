#!/bin/bash
# Qwen3.8-27B on an Apple Silicon Mac (M4 Pro, 24 GB) — llama.cpp on Metal.
#
# The vLLM stack in this repo is CUDA end to end, so on a Mac the same ideas are
# carried over to llama.cpp, which has native equivalents for the ones that matter
# at one stream:
#  - SPEC=dflash (default): the same DFlash2 block drafter the dflash2 arm uses
#    (z-lab / incoai Qwen3.8-27B-DFlash2, block size 8 = 7 drafts per pass), via
#    `--spec-type draft-dflash`. Exact: the target verifies every token.
#  - LOOKUP: drafting from the request's own context when the answer repeats it
#    (patches/dflash2-lookup-drafting.patch). llama.cpp's n-gram drafters do this
#    and take precedence over the model drafter whenever they fire, which is the
#    same "long block only while the copy runs" shape as LOOKUP_ADAPTIVE.
#  - SPEC=mtp: Qwen's own MTP head as a separate GGUF (unsloth MTP/), 3 drafts by
#    default — the count the 3090 arm settled on for long sessions.
#  - quantized KV (q8_0, the int8/fp8 KV of CTX=long) and the prompt cache with
#    recurrent-state checkpoints (PREFIX_CACHE=1 for a hybrid model).
# What does not carry over: Marlin/int8 tensor cores, the FA2/pth-flash-attn
# kernels, the 40k draft vocabulary (llama.cpp has no d2t path for qwen35 yet).
#
# Memory is the whole constraint on 24 GB of unified memory. The GPU may wire
# only ~2/3 of RAM by default (~16 GB here), and the IQ4_XS target alone is
# 14.25 GB, so the launcher refuses to start until `iogpu.wired_limit_mb` is
# raised (mac/README.md), when LM Studio has a model loaded, or when free memory
# does not cover the estimate. A watchdog kills the server if swap grows: the
# first MTP boot here went +590 MiB into swap and decoded at 6 tok/s against
# 15 once it fit, and a 27B in swap on this Mac has frozen it before.
#
#   bash mac/serve.sh                   # SPEC=dflash LOOKUP=ngram-mod CTX=fast (32k)
#   CHECK_ONLY=1 bash mac/serve.sh      # preflight only, loads nothing
#   SPEC=mtp CTX=long bash mac/serve.sh
#   MODEL_PRESET=flint bash mac/serve.sh   # Recursive Flint 40B-A2.5B MoE, see below
#
# MODEL_PRESET=flint swaps the target for Recursive Flint (RecursiveSI/flint,
# 40B total / 2.5B active MoE, unsloth UD-IQ3_XXS GGUF), a different model whose
# decode reads a fraction of the weights per token: the 60+ tok/s class on this
# Mac that no dense 27B reaches. There SPEC=dflash means incoai/Flint-DFlash2
# with 3 drafts (the default), plus lookup drafting as for Qwen.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/mac/common.sh"
LMS_MODELS="${LMS_MODELS:-$HOME/.lmstudio/models}"

MODEL_PRESET="${MODEL_PRESET:-qwen}"   # qwen | flint
# KV geometry per preset: full-attention layers, KV heads, head dim; and the
# per-sequence state besides the KV (Qwen3.8's Gated DeltaNet, ~150 MB in fp32).
case "$MODEL_PRESET" in
  qwen)  MODEL="${MODEL:-$LMS_MODELS/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf}"
         KV_LAYERS=16; KV_HEADS=4; HEAD_DIM=256; SEQ_STATE_MB=150 ;;
  # 14 of 49 layers are full attention; the other 35 slide over 513 tokens (~35 MiB in all)
  flint) MODEL="${MODEL:-$LMS_MODELS/unsloth/flint-GGUF/flint-UD-IQ3_XXS.gguf}"
         KV_LAYERS=14; KV_HEADS=4; HEAD_DIM=128; SEQ_STATE_MB=40
         # Flint has no MTP head. Its DFlash2 drafter (incoai/Flint-DFlash2, block 8) pays
         # only with a short block: a MoE verify wakes up to 6 experts per token per layer,
         # so 7 drafts cost more than they return (54.9 tok/s chat against 62.7 without),
         # and 3 is the best measured (65.8). 2026-10-30, M4 Pro, mac/results/.
         DFLASH_DRAFT="${DFLASH_DRAFT:-$LMS_MODELS/incoai/Flint-DFlash2-GGUF/Flint-DFlash2-Q8_0.gguf}"
         [ "${SPEC:-}" = mtp ] && { echo "Flint has no MTP head; use SPEC=dflash or off" >&2; exit 2; }
         SPEC="${SPEC:-dflash}"; DRAFT_TOKENS="${DRAFT_TOKENS:-3}"; CTX="${CTX:-long}" ;;
  *) echo "MODEL_PRESET must be qwen or flint (got '$MODEL_PRESET')" >&2; exit 2 ;;
esac
DFLASH_DRAFT="${DFLASH_DRAFT:-$LMS_MODELS/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
MTP_DRAFT="${MTP_DRAFT:-$LMS_MODELS/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"

SPEC="${SPEC:-dflash}"            # dflash | mtp | off
LOOKUP="${LOOKUP:-ngram-mod}"     # ngram-mod | ngram-simple | ngram-map-k | 0
CTX="${CTX:-fast}"                # fast = 32k, long = 64k, or a token count
KV_TYPE="${KV_TYPE:-q8_0}"        # f16 | q8_0 | q4_0 (quantized V needs -fa on)
NP="${NP:-1}"                     # server slots; one person chatting = 1
# The prompt cache and the checkpoints live in the same unified memory as the
# weights, and the cache fills up request by request: at 2048 MiB it pushed the
# MTP boot into swap halfway through the second benchmark request.
CACHE_RAM="${CACHE_RAM:-512}"     # MiB of prompt cache (llama.cpp default 8192 is a third of this Mac)
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-2}"  # recurrent-state snapshots per slot, ~150 MB each
DRAFT_UBATCH="${DRAFT_UBATCH:-64}"       # drafter batch; its 248k-vocab logits cost ~1 MiB per row
DRAFT_SAMPLING="${DRAFT_SAMPLING:-probabilistic}"  # the 3090 arm's draft_sample_method
PORT="${PORT:-18020}"
BIND="${BIND:-127.0.0.1}"
WATCHDOG="${WATCHDOG:-1}"
SWAP_LIMIT_MB="${SWAP_LIMIT_MB:-512}"    # swap growth that stops the server
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$CTX" in
  fast) CTX_TOKENS=32768 ;;
  long) CTX_TOKENS=65536 ;;
  *[!0-9]*|'') echo "CTX must be fast, long or a token count (got '$CTX')" >&2; exit 2 ;;
  *) CTX_TOKENS="$CTX" ;;
esac

case "$SPEC" in
  dflash) DRAFT="$DFLASH_DRAFT"; SPEC_TYPE=draft-dflash; DRAFT_TOKENS="${DRAFT_TOKENS:-7}" ;;
  mtp)    DRAFT="$MTP_DRAFT";    SPEC_TYPE=draft-mtp;    DRAFT_TOKENS="${DRAFT_TOKENS:-3}" ;;
  off)    DRAFT="";              SPEC_TYPE="" ;;
  *) echo "SPEC must be dflash, mtp or off (got '$SPEC')" >&2; exit 2 ;;
esac

# Lookup drafting. n-match 12 and a floor of 8 tokens mirror the 3090 arm's
# VLLM_DFLASH2_LOOKUP_NMAX=12 / _LONGMIN: the n-gram drafter only overrides the
# model drafter when it can propose a run long enough to be a copy. The ceiling
# is the verify block; on the M4 Pro a 16-token verify is already compute-bound,
# so it is lower than the 3090's 15 drafts + a long tail would suggest.
LOOKUP_MATCH="${LOOKUP_MATCH:-12}"
LOOKUP_MIN="${LOOKUP_MIN:-8}"
LOOKUP_MAX="${LOOKUP_MAX:-16}"
LOOKUP_ARGS=()
case "$LOOKUP" in
  0|off|'') LOOKUP="" ;;
  ngram-mod) LOOKUP_ARGS=(--spec-ngram-mod-n-match "$LOOKUP_MATCH" --spec-ngram-mod-n-min "$LOOKUP_MIN" --spec-ngram-mod-n-max "$LOOKUP_MAX") ;;
  ngram-simple) LOOKUP_ARGS=(--spec-ngram-simple-size-n "$LOOKUP_MATCH" --spec-ngram-simple-size-m "$LOOKUP_MAX") ;;
  ngram-map-k) LOOKUP_ARGS=(--spec-ngram-map-k-size-n "$LOOKUP_MATCH" --spec-ngram-map-k-size-m "$LOOKUP_MAX") ;;
  *) echo "LOOKUP must be ngram-mod, ngram-simple, ngram-map-k or 0 (got '$LOOKUP')" >&2; exit 2 ;;
esac

for f in "$MODEL" ${DRAFT:+"$DRAFT"}; do
  [ -f "$f" ] || { echo "missing $f — run: bash mac/download.sh${MODEL_PRESET:+ }$([ "$MODEL_PRESET" = flint ] && echo flint)" >&2; exit 1; }
done

# --- memory estimate --------------------------------------------------------
# KV: only the full-attention layers grow with the context (Qwen3.8: 16 of 64,
# the rest Gated DeltaNet; Flint: 14 of 49, the rest sliding-window), K and V.
# q8_0 stores 34 bytes per 32 values.
case "$KV_TYPE" in
  f16) KV_NUM=2; KV_DEN=1 ;; q8_0) KV_NUM=34; KV_DEN=32 ;; q4_0) KV_NUM=18; KV_DEN=32 ;;
  *) KV_NUM=2; KV_DEN=1 ;;
esac
MODEL_MB=$(mib "$MODEL")
DRAFT_MB=$([ -n "$DRAFT" ] && mib "$DRAFT" || echo 0)
KV_MB=$(( KV_LAYERS * 2 * KV_HEADS * HEAD_DIM * KV_NUM / KV_DEN * CTX_TOKENS * NP / 1048576 ))
STATE_MB=$(( SEQ_STATE_MB * NP * (1 + CTX_CHECKPOINTS) ))   # per-sequence state + checkpoints
COMPUTE_MB=1200                                     # Metal compute buffers, target
[ -n "$DRAFT" ] && COMPUTE_MB=$(( COMPUTE_MB + 400 + DRAFT_UBATCH ))   # + drafter context
NEED_MB=$(( MODEL_MB + DRAFT_MB + KV_MB + STATE_MB + COMPUTE_MB + CACHE_RAM ))
GPU_NEED_MB=$(( NEED_MB - CACHE_RAM - STATE_MB * CTX_CHECKPOINTS / (1 + CTX_CHECKPOINTS) ))

echo "== preflight"
echo "   target  $(basename "$MODEL")  ${MODEL_MB} MiB"
[ -n "$DRAFT" ] && echo "   drafter $(basename "$DRAFT")  ${DRAFT_MB} MiB  ($SPEC_TYPE, $DRAFT_TOKENS drafts)"
echo "   KV      $KV_TYPE x ${CTX_TOKENS} tokens x ${NP} slot(s)  ${KV_MB} MiB"
echo "   state   sequence + ${CTX_CHECKPOINTS} checkpoints  ${STATE_MB} MiB, prompt cache ${CACHE_RAM} MiB"
mac_preflight

# --- launch -----------------------------------------------------------------
source "$REPO/resolve_api_key.sh"
resolve_vllm_key
ARGS=(-m "$MODEL" --alias "$([ "$MODEL_PRESET" = flint ] && echo flint || echo qwen3.8-27b)" --host "$BIND" --port "$PORT"
      -ngl all -fa on -c "$CTX_TOKENS" -np "$NP"
      -ctk "$KV_TYPE" -ctv "$KV_TYPE"
      --cache-ram "$CACHE_RAM" --ctx-checkpoints "$CTX_CHECKPOINTS"
      --jinja --metrics)
TYPES="$LOOKUP"
[ -n "$SPEC_TYPE" ] && TYPES="${TYPES:+$TYPES,}$SPEC_TYPE"
[ -n "$TYPES" ] && ARGS+=(--spec-type "$TYPES")
[ -n "$DRAFT" ] && ARGS+=(-md "$DRAFT" -ngld all --spec-draft-n-max "$DRAFT_TOKENS"
                          --spec-draft-n-ubatch "$DRAFT_UBATCH" --spec-draft-sampling "$DRAFT_SAMPLING")
ARGS+=(${LOOKUP_ARGS[@]+"${LOOKUP_ARGS[@]}"})   # bash 3.2: an empty array trips set -u
[ -n "${VLLM_API_KEY:-}" ] && ARGS+=(--api-key "$VLLM_API_KEY")
# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

echo "== llama-server ${ARGS[*]}" | sed 's/--api-key [^ ]*/--api-key ***/'
mac_run_watched llama-server "${ARGS[@]}"
