#!/bin/bash
# Fetch the three GGUFs mac/serve.sh needs, into LM Studio's model folder so they
# stay out of this (cloud-synced) checkout and LM Studio lists them too,
#
#   target   unsloth/Qwen3.8-27B-GGUF  UD-IQ4_XS   14.25 GB (imatrix; fits beside a drafter in 24 GB)
#   dflash   incoai/Qwen3.8-27B-DFlash2-GGUF  Q4_K_M  1.14 GB
#   mtp      unsloth/Qwen3.8-27B-GGUF  MTP/ Q4_0   1.37 GB
#
# and, for mac/serve_mlx.sh (mlx_vlm), two MLX drafters kept outside LM Studio's
# folder because they are not chat models (the MLX target is LM Studio's own
# lmstudio-community/Qwen3.8-27B-MLX-4bit):
#
#   mlx-dflash  SSSAMMMM/Qwen3.8-27B-DFlash2-MLX-4bit   1.08 GB (4-bit of z-lab/Qwen3.8-27B-DFlash2)
#   mlx-mtp     mlx-community/Qwen3.8-27B-MTP-4bit      0.24 GB
#   flint       unsloth/flint-GGUF UD-IQ3_XXS   14.54 GB (Recursive Flint 40B-A2.5B MoE; MODEL_PRESET=flint)
#   flint-dflash incoai/Flint-DFlash2-GGUF Q8_0  0.43 GB (its DFlash2 drafter; MODEL_PRESET=flint SPEC=dflash)
#   flint-tempo9 thinkspread/Flint-Q6K-IQ3XXS-GGUF 18.03 GB (24 GB build for mac/serve_tempo9.sh)
#   mlx-text    text-only copy of LM Studio's lmstudio-community/Qwen3.8-27B-MLX-4bit
#               (mac/mlx_text_only.py: the vision tower out, 14.1 GiB, no download)
#
# The DFlash2 GGUF must be the 2026-08-29 revision or later: the release of
# 2026-08-18 lacks `dflash.rope.dimension_sections` (the "mrope fix"), and a
# drafter with the wrong RoPE still verifies exactly but accepts fewer tokens.
#
#   bash mac/download.sh            # all three
#   bash mac/download.sh target     # or any of: target dflash mtp flint flint-dflash flint-tempo9 mlx-dflash mlx-mtp mlx-text
set -euo pipefail

LMS_MODELS="${LMS_MODELS:-$HOME/.lmstudio/models}"
DRAFTERS="${DRAFTERS:-$HOME/models/mlx-drafters}"
QUANT="${QUANT:-UD-IQ4_XS}"
WHAT=("$@"); [ ${#WHAT[@]} -gt 0 ] || WHAT=(target dflash mtp)

get() {  # repo file dir
  mkdir -p "$3"
  hf download "$1" "$2" --local-dir "$3"
}

for w in "${WHAT[@]}"; do
  case "$w" in
    target) get unsloth/Qwen3.8-27B-GGUF "Qwen3.8-27B-$QUANT.gguf" "$LMS_MODELS/unsloth/Qwen3.8-27B-GGUF" ;;
    dflash) get incoai/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf "$LMS_MODELS/incoai/Qwen3.8-27B-DFlash2-GGUF" ;;
    mtp)    get unsloth/Qwen3.8-27B-GGUF MTP/mtp-Qwen3.8-27B-Q4_0.gguf "$LMS_MODELS/unsloth/Qwen3.8-27B-GGUF" ;;
    flint)  get unsloth/flint-GGUF flint-UD-IQ3_XXS.gguf "$LMS_MODELS/unsloth/flint-GGUF" ;;
    flint-dflash) get incoai/Flint-DFlash2-GGUF Flint-DFlash2-Q8_0.gguf "$LMS_MODELS/incoai/Flint-DFlash2-GGUF" ;;
    flint-tempo9) get thinkspread/Flint-Q6K-IQ3XXS-GGUF Flint-Q6K_IQ3XXS.gguf "$LMS_MODELS/thinkspread/Flint-Q6K-IQ3XXS-GGUF" ;;
    mlx-dflash) mkdir -p "$DRAFTERS"; hf download SSSAMMMM/Qwen3.8-27B-DFlash2-MLX-4bit --local-dir "$DRAFTERS/Qwen3.8-27B-DFlash2-MLX-4bit" ;;
    mlx-mtp)    mkdir -p "$DRAFTERS"; hf download mlx-community/Qwen3.8-27B-MTP-4bit --local-dir "$DRAFTERS/Qwen3.8-27B-MTP-4bit" ;;
    mlx-text)   python3 "$(dirname "$0")/mlx_text_only.py" "$LMS_MODELS/lmstudio-community/Qwen3.8-27B-MLX-4bit" "$HOME/models/mlx/Qwen3.8-27B-MLX-4bit-text" ;;
    *) echo "unknown: $w (target, dflash, mtp, flint, flint-dflash, flint-tempo9, mlx-dflash, mlx-mtp or mlx-text)" >&2; exit 2 ;;
  esac
done

D="$LMS_MODELS/incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
# the key sits after the tokenizer arrays, ~10.9 MB into the file; no grep -q,
# which closes the pipe early and fails the whole test under pipefail
if [ -f "$D" ] && ! head -c 32000000 "$D" | grep -a 'dflash.rope.dimension_sections' >/dev/null; then
  echo "!! $D predates the mrope fix — rerun: bash mac/download.sh dflash" >&2
fi
