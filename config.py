from dotenv import load_dotenv
from pathlib import Path
import os, queue

LLAMA_BASE = "http://localhost:8080"
VOICEVOX_BASE = "http://localhost:50021"
WHISPER_MODEL = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_CMD = "whisper {wav} --model medium --language ja --task transcribe --output_format txt --output_dir /tmp"
SYSTEM_PROMPT = "あなたは優しく簡潔に話すアシスタントです。返答は短く、敬体で。"
REC_SECONDS = 5
TIMEOUT = 2.0
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20
VAD_SENSITIVITY = 3
SILENCE_TAIL_MS = 500
LISTEN_ENABLED = True
INPUT_DEVICE = None
INPUT_CHANNELS = 2
CHANNEL_STRATEGY = "max"
PREFER_INPUT = ["USB Audio", "pulse", "pipewire", "default"]
WANTED_RATES = [16000, 48000]
AUDIO_Q = queue.Queue(maxsize=8)
WORKER_THREAD = None
VAD_THREAD = None
VOICEVOX_ID = 1


def load_config():
    global LLAMA_BASE, VOICEVOX_BASE, WHISPER_CMD, SYSTEM_PROMPT, REC_SECONDS
    global INPUT_DEVICE, INPUT_CHANNELS, CHANNEL_STRATEGY, PREFER_INPUT, WANTED_RATES, LISTEN_ENABLED
    load_dotenv()

    BASE_DIR = Path(__file__).resolve().parent

    LLAMA_BASE = os.getenv("LLAMA_BASE", "http://localhost:8080")
    VOICEVOX_BASE = os.getenv("VOICEVOX_BASE", "http://localhost:50021")
    WHISPER_CMD = os.getenv(
        "WHISPER_CMD",
        "whisper {wav} --model medium --language ja --task transcribe --output_format txt --output_dir /tmp",
    )

    SYSTEM_PROMPT_PATH = os.getenv(
        "SYSTEM_PROMPT_PATH",
        str(BASE_DIR / "SYSTEM_PROMPT.md"),
    )
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
    except FileNotFoundError:
        SYSTEM_PROMPT = "あなたは優しく簡潔に話すアシスタントです。返答は短く、敬体で。"

    VOICEVOX_ID = os.getenv("VOICEVOX_ID")
    REC_SECONDS = int(os.getenv("REC_SECONDS", str(REC_SECONDS)))
    INPUT_DEVICE = os.getenv("INPUT_DEVICE")
    INPUT_CHANNELS = int(os.getenv("INPUT_CHANNELS", str(INPUT_CHANNELS)))
    CHANNEL_STRATEGY = os.getenv("CHANNEL_STRATEGY", str(CHANNEL_STRATEGY))
    PREFER_INPUT = os.getenv("PREFER_INPUT", ",".join(PREFER_INPUT)).split(",")
    WANTED_RATES = [
        int(x)
        for x in os.getenv("WANTED_RATES", ",".join(map(str, WANTED_RATES))).split(",")
    ]
    LISTEN_ENABLED = os.getenv("LISTEN_ENABLED", "True").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
