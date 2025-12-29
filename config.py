from dotenv import load_dotenv
from pathlib import Path
import os, queue

LLAMA_BASE = "http://localhost:8080"
VOICEVOX_BASE = "http://localhost:50021"
VOICEVOX_SPEAKER_ID = 1
WHISPER_MODEL = "medium"
WHISPER_DEVICE = "cuda"
SYSTEM_PROMPT = "あなたは優しく簡潔に話すアシスタントです。返答は短く、敬体で。"
TIMEOUT = 2.0


def load_config():
    global LLAMA_BASE, VOICEVOX_BASE, VOICEVOX_SPEAKER_ID, SYSTEM_PROMPT
    load_dotenv()
    BASE_DIR = Path(__file__).resolve().parent

    LLAMA_BASE = os.getenv("LLAMA_BASE", "http://localhost:8080")
    VOICEVOX_BASE = os.getenv("VOICEVOX_BASE", "http://localhost:50021")
    VOICEVOX_SPEAKER_ID = int(
        os.getenv("VOICEVOX_SPEAKER_ID", str(VOICEVOX_SPEAKER_ID))
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
