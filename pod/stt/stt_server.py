"""
Minimal STT service wrapping faster-whisper, for the voice-agent pipeline.
Standalone HTTP endpoint, verified independently before Pipecat wires to it.
"""
import logging
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_debug")

MODEL_SIZE = "large-v3-turbo"
# Segments with no_speech_prob at or above this are silence/noise, not speech.
# Matches Pipecat's own WhisperSTTService default threshold.
NO_SPEECH_PROB_THRESHOLD = 0.4
DEBUG_AUDIO_DIR = "/workspace/voice-stack/logs/debug_audio"

import os
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

app = FastAPI(title="Voice Agent STT")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": "cuda"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # DEBUG: save every incoming utterance so we can inspect actual audio quality
    if os.getenv("DEBUG_AUDIO") == "1":
        ts = time.time()
        debug_path = f"{DEBUG_AUDIO_DIR}/{ts}.wav"
        with open(debug_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved debug audio: {debug_path} ({len(audio_bytes)} bytes)")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, info = model.transcribe(
            tmp.name,
            beam_size=5,
            # Pin English. Auto-detect guesses wrong on short/noisy utterances --
            # it decided a caller was speaking Spanish, so the LLM replied in Spanish.
            language="en",
            # faster-whisper's built-in Silero VAD. Drops silence before it reaches
            # the decoder, which is the main source of Whisper hallucinations.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Stops a bad transcript seeding the next one (repetition loops).
            condition_on_previous_text=False,
        )
        segments = list(segments)
        for seg in segments:
            logger.info(f"segment: no_speech_prob={seg.no_speech_prob:.3f} text={seg.text!r}")
        text = "".join(
            seg.text for seg in segments if seg.no_speech_prob < NO_SPEECH_PROB_THRESHOLD
        )
    return {"text": text.strip(), "language": info.language, "language_probability": info.language_probability}
