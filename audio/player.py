import sounddevice as sd
import soundfile as sf
import os
import config
import logging

logger = logging.getLogger(__name__)


def play_wav(wav_path: str):
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[player]録音データ: {wav_path} が存在しません")
    else:
        logger.info("Playing wav")
        OUTPUT_DEVICE = sd.query_devices(None, "output")["index"]
        signal, sampling_rate = sf.read(wav_path)
        sd.play(signal, sampling_rate, device=OUTPUT_DEVICE)
        sd.wait()
        return None
