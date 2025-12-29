import requests
import time
import config
import logging

logger = logging.getLogger(__name__)


def synthesize_to_wav(reply_text: str):
    logger.info("synthesize ready...")
    speaker_id = config.VOICEVOX_SPEAKER_ID  # VOICEVOXのボイスID
    q = requests.post(
        f"{config.VOICEVOX_BASE}/audio_query",
        params={"text": reply_text, "speaker": speaker_id},
        timeout=30,
    )
    q.raise_for_status()
    s = requests.post(
        f"{config.VOICEVOX_BASE}/synthesis",
        params={"speaker": speaker_id},
        json=q.json(),
        timeout=60,
    )
    s.raise_for_status()
    file_timestamp = int(time.time() * 1000)
    wav_path = "/tmp/reply_{file_timestamp}.wav"
    with open(wav_path, "wb") as f:
        f.write(s.content)
    return wav_path
    logger.info(f"wav_path:{wav_path}")
