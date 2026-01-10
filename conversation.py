from audio import recorder
from audio import player
from services import whisper_asr
from services import llama_client
from services import voicevox_tts
from storage.memory import now_jst_iso, init_db, save_message, load_recent_messages
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

            recent = load_recent_messages(limit=9, db_path="data/coco.db")

            system_prompt_stamped = f"現在日時:{now_jst_iso()}\n{config.SYSTEM_PROMPT}"
            messages = [{"role": "system", "content": system_prompt_stamped}]

            for role, content, created_at_jst in recent:
                content_stamped = f"{created_at_jst}\n{content}"
                messages.append({"role": role, "content": content_stamped})

            user_stamped = f"{now_jst_iso()}\n{text}"
            messages.append({"role": "user", "content": user_stamped})

            reply = llama_client.chat(messages)

            save_message("user", text, "data/coco.db")
            save_message("assistant", reply, "data/coco.db")
            logger.info(
                f"[DB]:{load_recent_messages(limit=10, db_path="data/coco.db")}"
            )

            wav_out = voicevox_tts.synthesize_to_wav(reply)
            player.play_wav(wav_out)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, exiting...")
    finally:
        pass
    return None


if __name__ == "__main__":
    run()
