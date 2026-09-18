"""
Voice-agent orchestrator: Twilio (or local webrtc for testing) <-> STT/LLM/TTS
(Deepgram STT + Rime TTS, hosted APIs -- no GPU pod required)
<-> odoo-tools for Odoo sales-order operations.
"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)  # must run before importing odoo_tools, which reads env vars at import time

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, STTMuteFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from odoo_tools import create_order, create_partner, get_order, get_orders, get_products

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
RIME_API_KEY = os.environ["RIME_API_KEY"]
RIME_VOICE_ID = os.getenv("RIME_VOICE_ID", "cove")

# LLM: Gemini via its OpenAI-compatible endpoint (personal key with existing
# Google Developer Program credit balance -- not a self-hosted model, so this
# removes the last GPU dependency too).
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")

MAX_CALL_DURATION_SECS = int(os.getenv("MAX_CALL_DURATION_SECS", "180"))

SYSTEM_INSTRUCTION = (
    "You are a phone sales assistant for AI Agentic Enterprises. "
    "Always reply in English, even if the caller's words appear to be in another "
    "language — transcription can misdetect language. Your responses "
    "will be spoken aloud, so avoid emojis, bullet points, or any formatting that "
    "can't be spoken. Keep replies brief and natural, like a real phone call. "
    "Use get_products to look up what's available before quoting prices. Use "
    "create_partner to register a new caller if they aren't an existing customer. "
    "Use create_order to place an order once you have a customer id, product id, "
    "and quantity. Use get_order or get_orders to check on existing orders. "
    "Never invent product ids, prices, or order numbers — always look them up."
)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting voice-agent bot")

    session = aiohttp.ClientSession()

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=None,  # defaults are fine; language pinned via settings below if needed
    )
    tts = RimeHttpTTSService(
        api_key=RIME_API_KEY,
        aiohttp_session=session,
        settings=RimeHttpTTSService.Settings(model="mistv2", voice=RIME_VOICE_ID),
    )

    from pipecat.services.openai.llm import OpenAILLMService

    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("One moment."))

    context = LLMContext(
        tools=[get_products, create_partner, create_order, get_order, get_orders]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8))
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    timeout_task: asyncio.Task | None = None

    async def _enforce_max_duration():
        await asyncio.sleep(MAX_CALL_DURATION_SECS)
        logger.info(f"Call reached {MAX_CALL_DURATION_SECS}s limit, wrapping up")
        # Mute STT so further speech can't interrupt/cancel the goodbye message
        # (Pipecat's normal barge-in behavior would otherwise discard it the
        # instant the caller starts talking again).
        await worker.queue_frames([STTMuteFrame(mute=True)])
        # Speak a fixed closing line directly instead of asking the LLM to
        # generate one — with an in-progress task (e.g. mid-collection of
        # account details) in context, the LLM tends to blend the wrap-up
        # instruction with the ongoing task instead of clearly ending the call.
        await tts.queue_frame(
            TTSSpeakFrame(
                "We have reached the two minute limit for this demo call. "
                "Thanks so much for trying our AI assistant, goodbye!"
            )
        )
        await asyncio.sleep(8)
        await worker.cancel()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal timeout_task
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the caller briefly and ask how you can help.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])
        timeout_task = asyncio.create_task(_enforce_max_duration())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if timeout_task:
            timeout_task.cancel()
        await worker.cancel()
        await session.close()

    from pipecat.workers.runner import WorkerRunner

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
