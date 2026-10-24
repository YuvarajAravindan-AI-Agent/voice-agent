"""
TTS provider switch. Pick the engine with TTS_PROVIDER so the rest of the
pipeline (Deepgram, Gemini LLM, Odoo tools) stays identical across A/B runs.

    rime-http  Rime over plain HTTP (the original setup)
    rime-ws    Rime over WebSocket (streaming, lower time to first audio)
    gemini     Gemini TTS via the Gemini API (uses an AI Studio API key)
"""
import os

import aiohttp

PROVIDERS = ("rime-http", "rime-ws", "gemini")


def build_tts(aiohttp_session: aiohttp.ClientSession | None = None):
    provider = os.getenv("TTS_PROVIDER", "rime-http").strip().lower()

    if provider in ("rime-http", "rime-ws"):
        api_key = os.environ["RIME_API_KEY"]
        model = os.getenv("RIME_MODEL", "mistv2")
        voice = os.getenv("RIME_VOICE_ID", "cove")
        if provider == "rime-http":
            from pipecat.services.rime.tts import RimeHttpTTSService

            if aiohttp_session is None:
                raise ValueError("rime-http needs an aiohttp session")
            return RimeHttpTTSService(
                api_key=api_key,
                aiohttp_session=aiohttp_session,
                settings=RimeHttpTTSService.Settings(model=model, voice=voice),
            )
        from pipecat.services.rime.tts import RimeTTSService

        return RimeTTSService(
            api_key=api_key,
            settings=RimeTTSService.Settings(model=model, voice=voice),
        )

    if provider == "gemini":
        from pipecat.services.google.tts import GeminiTTSService

        api_key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        return GeminiTTSService(
            api_key=api_key,
            settings=GeminiTTSService.Settings(
                model=os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
                voice=os.getenv("GEMINI_TTS_VOICE", "Kore"),
            ),
        )

    raise ValueError(f"Unknown TTS_PROVIDER {provider!r}; expected one of {PROVIDERS}")
