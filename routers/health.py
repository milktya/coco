from fastapi import APIRouter
import requests
import config

router = APIRouter()

@router.get("/health")
def health():
    result = {"llama": "down", "voicevox": "down", "ok": False}

    try:
        r = requests.get(f"{config.LLAMA_BASE}/v1/models", timeout=config.TIMEOUT)
        r.raise_for_status()
        result["llama"] = "up"
    except Exception as e:
        result["llama_error"] = str(e)[:160]

    try:
        r = requests.get(f"{config.VOICEVOX_BASE}/speakers", timeout=config.TIMEOUT)
        r.raise_for_status()
        result["voicevox"] = "up"
    except Exception as e:
        result["voicevox_error"] = str(e)[:160]

    result["ok"] = (result["llama"] == "up" and result["voicevox"] == "up")
    return result
