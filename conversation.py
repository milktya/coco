from audio import recorder
from audio import player
from services import whisper_asr
from services import llama_client
from services import voicevox_tts
from storage.memory import init_db, save_message, load_recent_messages
from logging_config import setup_logging
import config
import logging

config.load_config()

setup_logging()
logger = logging.getLogger(__name__)

init_db("coco.db")
SYSTEM_PROMPT = open("SYSTEM_PROMPT.md", "r", encoding="utf-8").read()


def run():
    try:
        while True:
            logger.info("conversation loop start (Ctrl+C to exit)")
            wav_in = recorder.record_audio()
            text = whisper_asr.transcribe(wav_in).strip()
            if not text:
                continue

            recent = load_recent_messages(limit=10, db_path="data/coco.db")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for role, content in recent:
                messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": text})  # 最後に今回入力を追加
            save_message("user", text, "data/coco.db")
            logger.info(f"[DB]:{messages}")

            reply = llama_client.chat(messages)

            save_message("assistant", reply, "data/coco.db")  # 返答も保存

            wav_out = voicevox_tts.synthesize_to_wav(reply)
            player.play_wav(wav_out)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, exiting...")
    finally:
        pass
    return None


if __name__ == "__main__":
    run()
