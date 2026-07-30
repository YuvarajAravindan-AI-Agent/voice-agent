#!/usr/bin/env bash
# Start the voice stack on a RunPod pod.
# Lives on the network volume so it survives pod termination:
#   bash /workspace/start-voice-stack.sh
set -uo pipefail

VS=/workspace/voice-stack
LOGS=$VS/logs
BIN=/workspace/bin

export OLLAMA_MODELS=/workspace/ollama-models
export OLLAMA_HOST=127.0.0.1:11434
# Keep the model resident. Default is 5m, after which qwen3 is evicted from VRAM
# and the next caller waits ~19s for a 5.2GB reload before hearing anything.
export OLLAMA_KEEP_ALIVE=-1
# Model caches on the VOLUME. Default ~/.cache is container disk, wiped on stop,
# which made STT/TTS re-download ~1.9GB on every start (~9 min cold start).
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch

mkdir -p "$LOGS" "$BIN" "$HF_HOME" "$TORCH_HOME"

# --- ollama: keep the FULL install on the volume ---
# Extracting only bin/ollama is NOT enough. Ollama needs lib/ollama (llama-server
# plus the CUDA runners) or it errors "llama-server binary not found" and falls
# back to CPU. Asset is .tar.zst on GitHub; ollama.com/download/*.tgz is a 404.
if [ ! -x "$BIN/ollama" ] || [ ! -x /workspace/lib/ollama/llama-server ]; then
  echo "[ollama] installing full runtime to /workspace (once, ~1.4GB)..."
  OLLAMA_VER=${OLLAMA_VER:-v0.32.5}
  curl -fL --retry 3 -o /tmp/ollama.tar.zst \
    "https://github.com/ollama/ollama/releases/download/${OLLAMA_VER}/ollama-linux-amd64.tar.zst" \
    && tar --zstd -xf /tmp/ollama.tar.zst -C /workspace \
    && rm -f /tmp/ollama.tar.zst
fi
export PATH="$BIN:$PATH"

start() {  # name port cmd...
  local name=$1 port=$2; shift 2
  if ss -tln 2>/dev/null | grep -q ":$port "; then
    echo "[$name] already listening on :$port"
    return
  fi
  echo "[$name] starting on :$port"
  setsid nohup "$@" >>"$LOGS/$name.log" 2>&1 < /dev/null &
}

start ollama 11434 "$BIN/ollama" serve
start stt 8010 "$VS/.venv/bin/uvicorn" --app-dir "$VS/stt" stt_server:app --host 0.0.0.0 --port 8010
start tts 8011 "$VS/.venv/bin/uvicorn" --app-dir "$VS/tts" tts_server:app --host 0.0.0.0 --port 8011

# --- qwen3-fast: qwen3:8b with reasoning permanently disabled ---
# Ollama's OpenAI-compatible /v1 endpoint (what Pipecat uses) never sets the
# $.IsThinkSet template variable, so stock qwen3 always reasons: ~1000 tokens and
# ~25s per reply, with the reasoning stripped from the response so you only see
# the latency. The stock template prefills an empty <think></think> block when
# thinking is off; making that unconditional forces it off on every endpoint.
# Result: 25.6s -> ~1.3s per reply.
for i in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 2; done
if ! "$BIN/ollama" list 2>/dev/null | grep -q '^qwen3-fast'; then
  echo "[ollama] building qwen3-fast (reasoning disabled)..."
  "$BIN/ollama" show --template qwen3:8b > /tmp/tmpl.txt 2>/dev/null
  python3 -c "
t=open('/tmp/tmpl.txt').read()
old='{{ if and \$.IsThinkSet (not \$.Think) -}}'
if old in t:
    open('/tmp/tmpl.txt','w').write(t.replace(old,'{{ if true -}}',1))
    print('  template patched')
else:
    raise SystemExit('  WARN: think guard not found - qwen3 template may have changed')
"
  { echo 'FROM qwen3:8b'; echo 'TEMPLATE """'; cat /tmp/tmpl.txt; echo '"""'; } > /tmp/Modelfile.nothink
  "$BIN/ollama" create qwen3-fast -f /tmp/Modelfile.nothink 2>&1 | tail -1
  rm -f /tmp/tmpl.txt /tmp/Modelfile.nothink
fi

echo
echo "waiting for services (first cold start downloads models, can take ~10 min)..."
for i in $(seq 1 240); do
  ok=0
  curl -sf -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && ok=$((ok+1))
  curl -sf -m 2 http://127.0.0.1:8010/health   >/dev/null 2>&1 && ok=$((ok+1))
  curl -sf -m 2 http://127.0.0.1:8011/health   >/dev/null 2>&1 && ok=$((ok+1))
  [ "$ok" -eq 3 ] && break
  sleep 3
done

# Pre-warm so the first real caller does not pay the model-load cost.
echo "pre-warming qwen3-fast into VRAM..."
curl -sf -m 300 http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3-fast","prompt":"hi","stream":false,"keep_alive":-1}' >/dev/null 2>&1

echo
echo "=== status ==="
printf "  ollama :11434  %s\n" "$(curl -sf -m 5 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && echo UP || echo DOWN)"
printf "  stt    :8010   %s\n" "$(curl -sf -m 5 http://127.0.0.1:8010/health   >/dev/null 2>&1 && echo UP || echo DOWN)"
printf "  tts    :8011   %s\n" "$(curl -sf -m 5 http://127.0.0.1:8011/health   >/dev/null 2>&1 && echo UP || echo DOWN)"
echo "  resident: $(curl -sf -m 5 http://127.0.0.1:11434/api/ps 2>/dev/null | python3 -c 'import sys,json;print(", ".join(m["name"] for m in json.load(sys.stdin).get("models",[])) or "none")' 2>/dev/null)"
echo
echo "Next: on contabo-tally update runpod-tunnel.service with this pod's IP/port,"
echo "then: systemctl restart runpod-tunnel"
