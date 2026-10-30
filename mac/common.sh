#!/bin/bash
# Shared by mac/serve.sh (llama.cpp) and mac/serve_mlx.sh (mlx_vlm): the memory
# preflight and the swap watchdog. Sourced; the caller sets NEED_MB (everything
# the server will hold) and GPU_NEED_MB (the part Metal has to wire) first.

mib() { echo $(( $(stat -f %z "$1") / 1048576 )); }
dir_mib() { echo $(( $(du -sk "$1" | cut -f1) / 1024 )); }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'; }

mac_preflight() {
  local ram_mb wired_mb free_pct free_mb fail=0 lms_loaded=""
  ram_mb=$(( $(sysctl -n hw.memsize) / 1048576 ))
  wired_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
  [ "$wired_mb" -gt 0 ] || wired_mb=$(( ram_mb * 2 / 3 ))   # macOS default for <= 36 GB
  free_pct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  free_mb=$(( ram_mb * ${free_pct:-0} / 100 ))
  echo "   total   ~${NEED_MB} MiB  (GPU-wired ~${GPU_NEED_MB} MiB)"
  echo "   Mac     ${ram_mb} MiB RAM, GPU wired limit ${wired_mb} MiB, free ~${free_mb} MiB (${free_pct:-?}%)"

  # `lms ps` starts the LM Studio service when it is not running, so ask only an
  # LM Studio that is already up; a closed one has nothing loaded.
  if pgrep -f 'LM Studio.app/Contents/MacOS/LM Studio' >/dev/null; then
    lms_loaded=$("$HOME/.lmstudio/bin/lms" ps 2>/dev/null | grep -v -e 'No models' -e 'lms load' -e '^[[:space:]]*$' -e 'To load' -e 'Waking up' || true)
  fi
  if [ -n "$lms_loaded" ]; then
    echo "!! LM Studio has a model loaded — unload it in LM Studio first (this script never does it for you):"
    echo "$lms_loaded" | sed 's/^/     /'
    fail=1
  fi
  # rec-mlx and tempo9 are matched by their bin/ path, so a shell or monitor
  # that only mentions them in its arguments does not count as a server.
  local servers='llama-server|mlx_lm.server|mlx_vlm.server|omlx-cli|omlx-server|oMLX.app/Contents/MacOS|bin/rec-mlx|bin/tempo9'
  if pgrep -f "$servers" >/dev/null; then
    echo "!! another local model server is running:"; pgrep -fl "$servers" | cut -c1-160 | sed 's/^/     /'
    fail=1
  fi
  if [ "$GPU_NEED_MB" -gt "$wired_mb" ]; then
    echo "!! the GPU may wire only ${wired_mb} MiB; raise it until the next reboot with:"
    echo "     sudo sysctl iogpu.wired_limit_mb=$(( (GPU_NEED_MB + 1024) / 512 * 512 ))"
    fail=1
  fi
  if [ "$(( NEED_MB + 1024 ))" -gt "$free_mb" ]; then
    echo "!! free memory (~${free_mb} MiB) does not cover ~${NEED_MB} MiB + 1 GiB margin. Largest processes:"
    ps -axmo rss=,comm= | head -6 | awk '{rss=$1; $1=""; printf "     %6d MiB %s\n", rss/1024, $0}'
    fail=1
  fi
  if [ "$fail" -ne 0 ] && [ "${FORCE:-0}" != 1 ]; then
    echo "refusing to start (FORCE=1 overrides at your own risk)"; exit 1
  fi
  echo "   ok"
  [ "${CHECK_ONLY:-0}" = 1 ] && exit 0
  return 0
}

# Run the server in the background and stop it if swap grows past
# SWAP_LIMIT_MB: a 27B that spills into swap on this Mac stops being usable long
# before it stops responding.
#
# LOAD_GRACE_S (default 0 = off) opens a window after launch in which swap may
# grow up to LOAD_GRACE_MB instead: MLX materializes the weights on the first
# request and pushes other processes out once. When the window closes the
# baseline is re-read and SWAP_LIMIT_MB applies from there, so a server that
# keeps swapping after its load is still stopped.
mac_run_watched() {
  "$@" &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' INT TERM EXIT
  if [ "${WATCHDOG:-1}" = 1 ]; then
    local swap0 growth limit started=$SECONDS grace=${LOAD_GRACE_S:-0}
    swap0=$(swap_mb)
    while kill -0 $SERVER 2>/dev/null; do
      sleep 5
      growth=$(( $(swap_mb) - swap0 ))
      if [ "$grace" -gt 0 ] && [ $(( SECONDS - started )) -lt "$grace" ]; then
        limit=${LOAD_GRACE_MB:-2048}
      else
        if [ "$grace" -gt 0 ]; then
          echo "== watchdog: load window closed with swap +${growth} MiB; limit ${SWAP_LIMIT_MB:-512} MiB from here" >&2
          swap0=$(swap_mb); growth=0; grace=0
        fi
        limit=${SWAP_LIMIT_MB:-512}
      fi
      if [ "$growth" -gt "$limit" ]; then
        echo "!! watchdog: swap grew ${growth} MiB (limit ${limit}) — stopping the server" >&2
        kill $SERVER; break
      fi
    done
  fi
  wait $SERVER
}
