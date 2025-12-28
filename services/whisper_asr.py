import whisper
import os
import shutil
import config

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading WhisperModel...")
        _model = whisper.load_model(
            config.WHISPER_MODEL, device=getattr(config, "WHISPER_DEVICE", "cpu")
        )


def transcribe(wav_path: str) -> str:
    # 録音データの有無を確認
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[whisper]録音データ: {wav_path} が存在しません")

    if shutil.which("ffmpeg") is None:
        print("[whisper]ffmegがインストールされていません")

    _model = get_model()
    result = model.transcribe(wav_path, language="ja", task="transcribe")
    text = (result.get["text"] or "").strip()
    print(f"transcribe_result: {text}")
    return text
