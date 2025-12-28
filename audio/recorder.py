import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import pyaudio
import wave
import time
import sounddevice as sd

# Silero VAD モデルのロード
print("Loading Silero VAD model...")
model = load_silero_vad()

# マイク入力の設定
CHUNK = 16000  # フレームサイズ (1秒分のデータ)
FORMAT = pyaudio.paInt16  # 音声フォーマット
CHANNELS = 1  # モノラル
RATE = 16000  # サンプリングレート
INPUT_DEVICE = sd.query_devices(None, "input")["index"]


def record_audio():
    # PyAudio ストリームを初期化
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=INPUT_DEVICE,
    )

    audio_buffer = []  # 音声データを一時的に保存する
    is_recording = False  # 現在有音かどうかを管理
    silence_count = 0  # 無音チャンク数のカウント
    tail_chunks = 2  # 許容する無音チャンク数

    try:
        while True:
            # マイクからデータを読み取り
            data = stream.read(CHUNK, exception_on_overflow=False)
            # バイナリデータをnumpy配列に変換
            audio_data = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            )
            # Silero VADで発話を判定
            speech = bool(get_speech_timestamps(audio_data, model, sampling_rate=RATE))

            if speech:
                # 音声が検出された場合
                if not is_recording:
                    print("[vad] ▶ start speech")
                    is_recording = True
                    audio_buffer = []  # バッファのクリア
                silence_count = 0  # 無音チャンク数カウントのクリア
                audio_buffer.append(data)  # 音声データをバッファに保存
            else:
                # 発話中の無音時
                if is_recording:
                    silence_count += 1
                    audio_buffer.append(data)  # 許容内の無音をバッファに追加
                    if silence_count > tail_chunks:
                        print("[vad] ■ end speech")
                        is_recording = False
                        silence_count = 0
                        full_audio = b"".join(audio_buffer)  # バッファを結合
                        audio_buffer = []  # バッファをクリア
                        # バッファをファイルに保存
                        file_timestamp = int(time.time() * 1000)
                        wav_path = f"/tmp/output_{file_timestamp}.wav"
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(audio.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(full_audio)
                        print(f"ファイルに保存しました: {wav_path}")
                        return wav_path
                else:
                    pass  # 無音時
    except KeyboardInterrupt:
        print("\n終了します...")
    finally:
        # ストリームとリソースを解放
        stream.stop_stream()
        stream.close()
        audio.terminate()
