"""
TTS service that calls the Kokoro HTTP server running on the RunPod GPU pod.
Mirrors pipecat's built-in PiperHttpTTSService pattern almost exactly, except
Kokoro's /speak endpoint takes the text as a GET query param and returns a
full WAV rather than a POST'd JSON body with chunked output.
"""
from collections.abc import AsyncGenerator

import aiohttp

from pipecat.frames.frames import ErrorFrame, Frame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroHttpTTSService(TTSService):
    """Calls a remote Kokoro /speak HTTP endpoint to synthesize speech."""

    def __init__(self, *, base_url: str, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            async with self._session.get(
                f"{self._base_url}/speak", params={"text": text}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(error=f"Kokoro TTS error {response.status}: {error}")
                    yield TTSStoppedFrame(context_id=context_id)
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
