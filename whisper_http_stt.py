"""
STT service that calls the faster-whisper HTTP server running on the RunPod
GPU pod, instead of loading a model in-process (this orchestrator runs on a
CPU-only VPS). Mirrors pipecat's built-in SegmentedSTTService contract: it
buffers one VAD-detected utterance as a WAV container and calls run_stt once
per utterance.
"""
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class WhisperHttpSTTService(SegmentedSTTService):
    """Calls a remote faster-whisper /transcribe HTTP endpoint per utterance."""

    def __init__(self, *, base_url: str, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            form = aiohttp.FormData()
            form.add_field("file", audio, filename="segment.wav", content_type="audio/wav")
            async with self._session.post(f"{self._base_url}/transcribe", data=form) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(error=f"STT error {response.status}: {error}")
                    return
                result = await response.json()
                text = (result.get("text") or "").strip()
                if text:
                    logger.debug(f"Transcription: [{text}]")
                    yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
        except Exception as e:
            yield ErrorFrame(error=f"Unknown STT error: {e}")
