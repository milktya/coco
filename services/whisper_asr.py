import whisper
import os
import shutil
import config
import logging

logger = logging.getLogger(__name__)
_model = None


def _get_model():
    logger.info("whisper ready...")
    global _model
    if _model is None:
        logger.info("Loading WhisperModel...")
        try:
            _model = whisper.load_model(
                config.WHISPER_MODEL, device=getattr(config, "WHISPER_DEVICE", "cpu")
            )
        except Exception as e:
            logger.info(f"[whisper] load_model failed: {e!r}")
            raise
    return _model


def transcribe(wav_path: str) -> str:
    # 録音データの有無を確認
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[whisper]録音データ: {wav_path} が存在しません")

    if shutil.which("ffmpeg") is None:
        logger.info("[whisper]ffmegがインストールされていません")

    model = _get_model()
    if model is None:
        raise RuntimeError("[whisper] model is None (load_model failed silently)")

    result = model.transcribe(wav_path, language="ja", task="transcribe")
    text = (result.get("text") or "").strip()
    logger.info(f"transcribe_result: {text}")
    return text
