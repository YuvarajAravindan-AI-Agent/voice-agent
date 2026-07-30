# Pod side of the voice stack

These run on the **RunPod GPU pod**, from the network volume at `/workspace`.
The files at the repo root are the *client* side (Pipecat services on
contabo-tally that call these over the SSH tunnel).

| Repo path | Runs at |
| --- | --- |
| `pod/start-voice-stack.sh` | `/workspace/start-voice-stack.sh` |
| `pod/stt/stt_server.py`    | `/workspace/voice-stack/stt/stt_server.py` |
| `pod/tts/tts_server.py`    | `/workspace/voice-stack/tts/tts_server.py` |

## Bringing a pod up

1. Deploy a pod with `amused_black_squirrel_volume` attached (EU-RO-1).
2. `bash /workspace/start-voice-stack.sh`
3. On contabo-tally, point `runpod-tunnel.service` at the new IP/port and
   `systemctl restart runpod-tunnel`.

## Gotchas these files encode

- **Ollama needs its full runtime**, not just `bin/ollama`. Without
  `lib/ollama/` (llama-server + CUDA runners) it errors "llama-server binary
  not found" and falls back to CPU. Asset is `.tar.zst` on GitHub releases;
  `ollama.com/download/*.tgz` is a 404.
- **`HF_HOME`/`TORCH_HOME` must point at `/workspace/.cache`.** The default
  `~/.cache` is container disk and is wiped on stop, causing a ~1.9GB
  re-download and ~9 minute cold start every time.
- **`OLLAMA_KEEP_ALIVE=-1`.** Default 5m eviction means the next caller waits
  ~19s for a 5.2GB VRAM reload.
- **`qwen3-fast`.** Ollama's OpenAI-compatible `/v1` endpoint (which Pipecat
  uses) never sets the `$.IsThinkSet` template variable, so stock qwen3 always
  reasons: ~1000 tokens and ~25s per reply, with the reasoning stripped from the
  response so only the latency is visible. `/no_think` in system or user
  messages does not work on `/v1`. The fix is a derived model whose template
  prefills an empty `<think></think>` unconditionally. 25.6s -> ~1.3s.
- **STT pins `language="en"`.** Auto-detect misread a caller as Spanish and the
  LLM replied in Spanish. `vad_filter=True` also suppresses Whisper's
  hallucinations on silence.
- **Debug audio is off by default.** Set `DEBUG_AUDIO=1` to record caller
  utterances; it writes every utterance to the volume forever otherwise.
