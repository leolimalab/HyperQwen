#!/bin/bash
# Recursive Flint on rec's own MLX engine (resident Metal megakernel, expert
# and KV kernels), for comparison with MODEL_PRESET=flint in mac/serve.sh
# (llama.cpp, 62-66 tok/s on this Mac).
#
# The official rec refuses Macs under 48 GiB and its checkpoint is 22.6 GB.
# This runs the engine of the fork thinkspread/rec@local/flint-e3-24gb, which
# drops the gate and accepts 3-bit routed experts, on a checkpoint made with
# mac/rec_requant_e3.py (~17.8 GB). The engine allocates the KV in fixed
# blocks, 65,536 tokens upstream (1.78 GiB for the first, even for a short
# context). With that block the first request pushed 3 GB into swap on this
# Mac, so mac/rec-e3-kv16k.patch cuts the block to 16,384 tokens (0.48 GiB; up
# to 4 blocks, 65,536 tokens of context) and turns the allocator cache off.
#
# Setup, outside the Drive:
#   git clone --branch local/flint-e3-24gb https://github.com/thinkspread/rec.git ~/src/rec-e3
#   git -C ~/src/rec-e3 apply "$PWD/mac/rec-e3-kv16k.patch"
#   uv venv ~/.venvs/rec-e3 --python 3.12
#   uv pip install --python ~/.venvs/rec-e3/bin/python ~/src/rec-e3/engine
#   hf download RecursiveSI/flint-int4 --local-dir ~/models/flint-int4
#   ~/.venvs/rec-e3/bin/python -I mac/rec_requant_e3.py ~/models/flint-int4 ~/models/flint-int4-e3 --delete-src
#
#   CHECK_ONLY=1 bash mac/serve_rec.sh
#   LOAD_GRACE_S=120 bash mac/serve_rec.sh      # model id "flint" on :18020
#   python3 mac/bench.py <label> --model flint --stream --quote-max-tokens 8000
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/mac/common.sh"

VENV="${VENV:-$HOME/.venvs/rec-e3}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/flint-int4-e3}"
MANIFEST="${MANIFEST:-$MODEL_DIR/flint-int4-e3.json}"
CTX="${CTX:-16384}"            # --context-limit, prompt and reply together
PORT="${PORT:-18020}"
METRICS="${METRICS:-}"         # per-request engine metrics (JSONL), off by default
WATCHDOG="${WATCHDOG:-1}"
SWAP_LIMIT_MB="${SWAP_LIMIT_MB:-256}"   # MLX: the weights are materialized and wired at load
EXTRA_ARGS="${EXTRA_ARGS:-}"
REC_MLX="$VENV/bin/rec-mlx"

[ -x "$REC_MLX" ] || { echo "missing $REC_MLX — see the setup lines at the top of this script" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST — run mac/rec_requant_e3.py first" >&2; exit 1; }

# --- memory estimate --------------------------------------------------------
# Weights, plus the KV blocks the context needs (14 full-attention layers x
# block tokens x 4 heads x 128 x K/V in BF16: 1792 MiB per 65,536 tokens; the
# first also holds the 35 sliding layers' 513 tokens, 35 MiB), plus workspace.
MODEL_MB=0
for f in "$MODEL_DIR"/model-*.safetensors; do MODEL_MB=$(( MODEL_MB + $(mib "$f") )); done
BLOCK_TOKENS=$("$VENV/bin/python" -c 'import rec_mlx.cache as c; print(c.BLOCK_TOKENS)')
BLOCKS=$(( (CTX + BLOCK_TOKENS - 1) / BLOCK_TOKENS ))
KV_MB=$(( BLOCKS * BLOCK_TOKENS * 14 * 1024 * 2 / 1048576 + 35 ))
COMPUTE_MB=800
NEED_MB=$(( MODEL_MB + KV_MB + COMPUTE_MB ))
GPU_NEED_MB=$NEED_MB

echo "== preflight (rec-mlx $(cd "$HOME/src/rec-e3" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo '?'), mlx $("$VENV/bin/python" -c 'import mlx.core as mx; print(mx.__version__)'))"
echo "   target  $(basename "$MODEL_DIR")  ${MODEL_MB} MiB  ($(python3 -c 'import json,sys; e=json.load(open(sys.argv[1]))["mlx_export"]; print("experts INT%d, dense INT%d" % (e["routed_expert_bits"], e["bits"]))' "$MODEL_DIR/config.json"))"
echo "   KV      BF16, ${BLOCKS} block(s) of ${BLOCK_TOKENS} tokens for ${CTX}  ${KV_MB} MiB"
mac_preflight

# --- launch -----------------------------------------------------------------
# rec-mlx reads its key from a file; with no key anywhere it binds none (it
# listens on loopback only).
source "$REPO/resolve_api_key.sh"
resolve_vllm_key
ARGS=(serve flint --model-dir "$MODEL_DIR" --manifest "$MANIFEST" --context-limit "$CTX" --port "$PORT")
if [ -n "${VLLM_API_KEY:-}" ]; then
  KEY_FILE=$(mktemp -t rec-key)
  chmod 600 "$KEY_FILE"
  printf '%s\n' "$VLLM_API_KEY" > "$KEY_FILE"
  ARGS+=(--api-key-file "$KEY_FILE")
fi
[ -n "$METRICS" ] && ARGS+=(--metrics "$METRICS")
# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

echo "== rec-mlx ${ARGS[*]}"
mac_run_watched "$REC_MLX" "${ARGS[@]}"
if [ -n "${KEY_FILE:-}" ]; then rm -f "$KEY_FILE"; fi
