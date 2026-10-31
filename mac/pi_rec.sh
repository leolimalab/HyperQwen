#!/bin/bash
# Pi (pi.dev, the coding agent Recursive pairs Flint with) against the rec
# server of mac/serve_rec.sh. It does what `rec launch pi` does, without the
# rec CLI (which refuses Macs under 48 GiB): Pi pinned by rec's package-lock,
# plus rec's own Pi extension, which registers the `rec` provider on Pi's
# OpenAI transport and always asks for the remaining context as max_tokens.
#
# Everything lives outside the Drive, in PI_HOME (default
# ~/.local/share/hyperqwen-pi): npm/ holds Pi, extension/ the rec extension,
# agent/ Pi's settings and sessions (kept apart from any Pi of your own).
#
# Start the server first, in another terminal (it needs ~19 GB; see
# mac/serve_rec.sh), then:
#
#   INSTALL_ONLY=1 bash mac/pi_rec.sh      # install Pi and the extension, nothing else
#   bash mac/pi_rec.sh                     # Pi in this terminal, in the current folder
#   bash mac/pi_rec.sh -p "list the files" # any Pi arguments pass through
#
# Pi works on the folder it starts in, with tools that read, write and run
# commands there. The context is the server's (CTX, 16k by default), and Pi's
# system prompt and tools take a good part of it.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REC_SRC="${REC_SRC:-$HOME/src/rec-e3}"
PI_HOME="${PI_HOME:-$HOME/.local/share/hyperqwen-pi}"
PORT="${PORT:-18020}"
CTX="${CTX:-16384}"            # must match the server's --context-limit
BASE_URL="http://127.0.0.1:${PORT}/v1"
CLIENT="$REC_SRC/cli/clients"

for f in pi/package.json pi/package-lock.json pi/flint-pi.ts pi/pi-transport.mjs pi/pi-logo.mjs opencode/recursive-logo/pixels.json; do
  [ -f "$CLIENT/$f" ] || { echo "missing $CLIENT/$f — clone the rec fork first (see mac/serve_rec.sh)" >&2; exit 1; }
done
command -v npm >/dev/null || { echo "Pi needs Node.js 22 and npm" >&2; exit 1; }

# --- Pi, pinned by rec's lockfile ------------------------------------------
PI="$PI_HOME/npm/node_modules/.bin/pi"
if [ ! -x "$PI" ] || ! cmp -s "$CLIENT/pi/package-lock.json" "$PI_HOME/npm/package-lock.json"; then
  echo "== installing $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["dependencies"]; print(", ".join("%s@%s" % kv for kv in d.items()))' "$CLIENT/pi/package.json") into $PI_HOME/npm"
  rm -rf "$PI_HOME/npm"
  mkdir -p "$PI_HOME/npm"
  cp "$CLIENT/pi/package.json" "$CLIENT/pi/package-lock.json" "$PI_HOME/npm/"
  (cd "$PI_HOME/npm" && npm ci --no-audit --no-fund --loglevel=error)
fi

# --- rec's extension, refreshed from the fork on every run -------------------
mkdir -p "$PI_HOME/extension" "$PI_HOME/agent"
cp "$CLIENT/pi/flint-pi.ts" "$CLIENT/pi/pi-transport.mjs" "$CLIENT/pi/pi-logo.mjs" "$PI_HOME/extension/"
cp "$CLIENT/opencode/recursive-logo/pixels.json" "$PI_HOME/extension/logo-pixels.json"
# The extension imports @mariozechner/pi-ai; resolve it from Pi's own install.
ln -sfn "$PI_HOME/npm/node_modules" "$PI_HOME/extension/node_modules"
chmod 700 "$PI_HOME"

echo "== pi $("$PI" --version 2>/dev/null || echo '?') at $PI"
[ "${INSTALL_ONLY:-0}" = 1 ] && exit 0

# --- the server ---------------------------------------------------------------
# Same key the launcher bound, if any; a server with no key accepts any value,
# and the extension refuses an empty one.
source "$REPO/resolve_api_key.sh"
resolve_client_key
KEY="$OPENAI_API_KEY"
if ! curl -fsS -m 5 -H "Authorization: Bearer $KEY" "$BASE_URL/models" >/dev/null 2>&1; then
  echo "no rec server on $BASE_URL — start it first: SWAP_LIMIT_MB=512 LOAD_GRACE_S=60 LOAD_GRACE_MB=1024 bash mac/serve_rec.sh" >&2
  exit 1
fi

echo "== Pi -> rec/flint at $BASE_URL, context $CTX, in $(pwd)"
export REC_API_KEY="$KEY"
export REC_PI_CONFIG="{\"baseUrl\":\"$BASE_URL\",\"contextWindow\":$CTX,\"maxTokens\":$CTX}"
export PI_CODING_AGENT_DIR="$PI_HOME/agent" PI_SKIP_VERSION_CHECK=1
unset PI_CODING_AGENT_SESSION_DIR
exec "$PI" --extension "$PI_HOME/extension/flint-pi.ts" --provider rec --model flint "$@"
