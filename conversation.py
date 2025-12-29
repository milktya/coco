from audio import recorder
from audio import player
from services import whisper_asr
from services import llama_client
from services import voicevox_tts
from logging_config import setup_logging
import config
import logging

config.load_config()

setup_logging()
logger = logging.getLogger(__name__)


def run():
    try:
        while True:
            logger.info("conversation loop start (Ctrl+C to exit)")
            wav_in = recorder.record_audio()
            text = whisper_asr.transcribe(wav_in).strip()
            if not text:
                continue

            reply = llama_client.chat(text)
            wav_out = voicevox_tts.synthesize_to_wav(reply)
            player.play_wav(wav_out)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, exiting...")
    finally:
        pass
    return None


if __name__ == "__main__":
    run()
