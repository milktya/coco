from audio import recorder
from audio import player
from services import whisper_asr
from services import llama_client
from services import voicevox_tts


def run():
    wav_in = recorder.record_audio()
    text = whisper_asr.transcribe(wav_in)

    # if not text.strip():
    #     continue

    reply = llama_client.chat(text)
    wav_out = voicevox_tts.speak(reply)
    player.play_spreak(wav_out)

    return None
