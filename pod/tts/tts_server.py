"""
Minimal TTS service wrapping Kokoro-82M, for the voice-agent pipeline.
Standalone HTTP endpoint, verified independently before Pipecat wires to it.
"""
import io

import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro import KPipeline

app = FastAPI(title="Voice Agent TTS")
pipeline = KPipeline(lang_code="a")  # "a" = American English
VOICE = "af_heart"


@app.get("/health")
def health():
    return {"status": "ok", "voice": VOICE}


@app.get("/speak")
def speak(text: str):
    generator = pipeline(text, voice=VOICE)
    audio_chunks = [audio for _, _, audio in generator]
    import numpy as np
    audio = np.concatenate(audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")
