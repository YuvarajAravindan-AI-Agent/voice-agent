"""
Render the same test sentences through each TTS provider and save the audio
so voices and latency can be compared without a phone call.

    python compare_tts.py                 # every provider whose key is set
    python compare_tts.py rime gemini     # only these

Output: out/<provider>/NN.<ext> plus out/results.csv (seconds to full audio).
This calls the vendors' REST APIs directly, so it measures total synthesis
time, not the streaming time-to-first-audio the live bot gets from rime-ws.
"""
import csv
import os
import sys
import time
import wave
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SENTENCES = [
    "Hello, thanks for calling AI Agentic Enterprises. How can I help you today?",
    "Your order S O 0 0 4 2 7 for 12 units has been placed.",
    "That comes to four thousand two hundred and fifty rupees, including tax.",
    "One moment, let me check the stock for that product.",
    "I'm sorry, I couldn't find a customer with that phone number.",
    "We have reached the two minute limit for this demo call. Thanks, goodbye!",
]

OUT = Path("out")


def rime(text: str, dest: Path) -> Path:
    r = requests.post(
        "https://users.rime.ai/v1/rime-tts",
        headers={
            "Authorization": f"Bearer {os.environ['RIME_API_KEY']}",
            "Accept": "audio/mp3",
        },
        json={
            "text": text,
            "speaker": os.getenv("RIME_VOICE_ID", "cove"),
            "modelId": os.getenv("RIME_MODEL", "mistv2"),
        },
        timeout=60,
    )
    r.raise_for_status()
    path = dest.with_suffix(".mp3")
    path.write_bytes(r.content)
    return path


def gemini(text: str, dest: Path) -> Path:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=os.getenv("GEMINI_TTS_VOICE", "Kore")
                    )
                )
            ),
        ),
    )
    pcm = resp.candidates[0].content.parts[0].inline_data.data
    path = dest.with_suffix(".wav")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return path


def gemini_cloud(text: str, dest: Path) -> Path:
    """Gemini TTS through Cloud Text-to-Speech, using the gcloud login (no API key)."""
    import base64
    import subprocess

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True
    ).stdout.strip()
    r = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": os.environ["GOOGLE_CLOUD_PROJECT"],
        },
        json={
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": os.getenv("GEMINI_TTS_VOICE", "Kore"),
                "modelName": os.getenv("GEMINI_CLOUD_TTS_MODEL", "gemini-2.5-flash-tts"),
            },
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
        },
        timeout=60,
    )
    r.raise_for_status()
    path = dest.with_suffix(".wav")
    path.write_bytes(base64.b64decode(r.json()["audioContent"]))
    return path


PROVIDERS = {
    "rime": (rime, "RIME_API_KEY"),
    "gemini": (gemini, ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "gemini-cloud": (gemini_cloud, "GOOGLE_CLOUD_PROJECT"),
}


def has_key(names) -> bool:
    names = (names,) if isinstance(names, str) else names
    return any(os.getenv(n) for n in names)


def main() -> None:
    wanted = sys.argv[1:] or list(PROVIDERS)
    rows = []
    for name in wanted:
        fn, keys = PROVIDERS[name]
        if not has_key(keys):
            print(f"skip {name}: no API key set")
            continue
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(SENTENCES, 1):
            start = time.perf_counter()
            path = fn(text, folder / f"{i:02d}")
            secs = time.perf_counter() - start
            rows.append((name, i, round(secs, 2), str(path)))
            print(f"{name} #{i}: {secs:.2f}s -> {path}")
    if rows:
        with open(OUT / "results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["provider", "sentence", "seconds", "file"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
