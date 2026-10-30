#!/bin/bash
# Recursive Flint on Tempo9 (ThinkSpread's Apple Silicon engine), for
# comparison with MODEL_PRESET=flint in mac/serve.sh (llama.cpp).
#
# Recursive's own engine (rec) refuses Macs under 48 GiB. The rec fork
# thinkspread/rec@local/flint-e3-24gb lifts that gate but needs an INT3-expert
# MLX checkpoint nobody publishes; the same authors ship Tempo9 (Flint since
# 1.2.0) and a 24 GB build of Flint as a GGUF, thinkspread/Flint-Q6K-IQ3XXS-GGUF
# (Q6_K dense layers, IQ3_XXS/IQ4_XS routed experts, 18.0 GB, top-1 agreement
# with Q8_0 85.3 % against the official INT4's 85.1 %). Their published number
# for the speed-only Q4_K-dense sibling is 123-127 tok/s on an M5 Pro 24 GB.
#
# NOT USEFUL ON AN M4 (measured 2026-10-30 on the M4 Pro 24 GB): Tempo9's Metal
# backend needs the Metal 4 tensor API of M5/M6/A19/A20 and has "no pre-M5
# fallback by design", so on an M4 it runs on the CPU (~17 tok/s on Flint).
# AS_METAL_TENSOR_FORCE=1 turns the GPU path on anyway and it produces garbage
# (~67 tok/s of nonsense). The script stays for an M5 Mac.
#
# Tempo9: `brew install thinkspread/tap/tempo9` (macOS 26). Its CLI and SDK are
# Apache-2.0; the engine is a closed binary, free for individuals. The server
# listens on loopback only and has no authentication.
#
#   CHECK_ONLY=1 bash mac/serve_tempo9.sh
#   bash mac/serve_tempo9.sh        # model id "flint" on :18020
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/mac/common.sh"
LMS_MODELS="${LMS_MODELS:-$HOME/.lmstudio/models}"

MODEL="${MODEL:-$LMS_MODELS/thinkspread/Flint-Q6K-IQ3XXS-GGUF/Flint-Q6K_IQ3XXS.gguf}"
NAME="${NAME:-flint}"
CTX="${CTX:-32768}"            # --max-length, prompt and reply together
MAX_BATCH="${MAX_BATCH:-1}"    # single-stream benchmark; Tempo9's default is 16
PORT="${PORT:-18020}"
WATCHDOG="${WATCHDOG:-1}"
SWAP_LIMIT_MB="${SWAP_LIMIT_MB:-512}"
TEMPO9="${TEMPO9:-$(command -v tempo9 || echo /opt/homebrew/bin/tempo9)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[ -x "$TEMPO9" ] || { echo "tempo9 not found — install it with: brew install thinkspread/tap/tempo9" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "missing $MODEL — run: bash mac/download.sh flint-tempo9" >&2; exit 1; }

# --- memory estimate --------------------------------------------------------
# Paged KV allocates for the tokens in use (Flint: 14 full-attention layers x
# 4 KV heads x 128, the other 35 slide over 513 tokens), so the budget is the
# weights plus a full context of fp16 KV plus compute buffers.
MODEL_MB=$(mib "$MODEL")
KV_MB=$(( 14 * 2 * 4 * 128 * 2 * CTX * MAX_BATCH / 1048576 ))
COMPUTE_MB=1000
NEED_MB=$(( MODEL_MB + KV_MB + COMPUTE_MB ))
GPU_NEED_MB=$NEED_MB

echo "== preflight (tempo9 $(brew list --versions tempo9 2>/dev/null | cut -d" " -f2))"
echo "   target  $(basename "$MODEL")  ${MODEL_MB} MiB"
echo "   KV      fp16 x ${CTX} tokens x ${MAX_BATCH} request(s), paged  ${KV_MB} MiB at most"
mac_preflight

ARGS=(--gguf "$MODEL" --name "$NAME" --port "$PORT" --max-length "$CTX" --max-batch "$MAX_BATCH")
# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)
echo "== tempo9 ${ARGS[*]}"
mac_run_watched "$TEMPO9" "${ARGS[@]}"
