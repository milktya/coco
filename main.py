from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import math
import os
import requests
import shlex
import subprocess
import time
import wave
import logging
import threading, queue, numpy as np, sounddevice as sd, webrtcvad
import queue, threading

load_dotenv()
LLAMA_BASE = os.getenv("LLAMA_BASE", "http://localhost:8080")
VOICEVOX_BASE = os.getenv("VOICEVOX_BASE", "http://localhost:50021")
SYSTEM_PROMPT = "あなたは優しく簡潔に話すアシスタントです。返答は短く、敬体で。"
REC_SECONDS = int(os.getenv("REC_SECONDS" , 5))
TIMEOUT = 2.0
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20              # 10/20/30のいずれか
VAD_SENSITIVITY = 3            # 0(ゆるめ)〜3(厳しめ)
SILENCE_TAIL_MS = 500          # 終端とみなす無音継続時間
LISTEN_ENABLED = True          # 起動と同時に監視する場合はTrue
INPUT_DEVICE = os.getenv("INPUT_DEVICE")
INPUT_CHANNELS = int(os.getenv("INPUT_CHANNELS", "2"))  # 2chで取って
CHANNEL_STRATEGY = os.getenv("CHANNEL_STRATEGY", "max") # "max" / "mean" / "left" / "right"
PREFER_INPUT = os.getenv("PREFER_INPUT", "UAB-80,USB Audio,pulse,pipewire,default").split(",")
WANTED_RATES = [int(x) for x in os.getenv("WANTED_RATES", "16000,48000").split(",")]
AUDIO_Q = queue.Queue(maxsize=8)
WORKER_THREAD = None
VAD_THREAD = None

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def vad_worker():
    while True:
        audio = AUDIO_Q.get()
        if audio is None:
            break
        try:
            # --- ここが今までcallback内にあった重い処理 ---
            stamp = int(time.time() * 1000)
            path = f"/tmp/vad_{stamp}.wav"
            write_wav(path, audio)
            logging.info(f"[vad] saved segment: {path}, samples={len(audio)}")

            whisper_cmd = os.getenv("WHISPER_CMD")
            cmd = whisper_cmd.format(wav=path, txtbase=f"/tmp/vad_{stamp}")
            logging.info(f"[vad] whisper cmd: {cmd}")
            _run(cmd, timeout=300)

            txt = f"/tmp/vad_{stamp}.txt"
            if os.path.exists(txt):
                with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
            else:
                alt = f"/tmp/vad_{stamp}.wav.txt"
                text = open(alt, "r", encoding="utf-8", errors="ignore").read().strip() if os.path.exists(alt) else ""

            if text:
                r = requests.post("http://localhost:8000/transcripts", json={"text": text}, timeout=120)
                r.raise_for_status()
                logging.info(f"[vad] heard: {text[:60]}")
        except Exception as e:
            logging.error(f"[vad] worker error: {e}")
        finally:
            AUDIO_Q.task_done()

def dbfs(x_int16: np.ndarray) -> float:
    x = x_int16.astype(np.float32) / 32768.0
    rms = math.sqrt((x*x).mean() + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def _pick_input_device(req_ch: int):
    devs = sd.query_devices()
    cand = [(i, d) for i, d in enumerate(devs) if (d.get("max_input_channels", 0) or 0) >= 1]

    def score(name: str) -> int:
        for k, key in enumerate(PREFER_INPUT):
            if key and key in name:
                return k
        return len(PREFER_INPUT)
    cand.sort(key=lambda x: score(x[1]["name"]))

    for idx, d in cand:
        max_in = d.get("max_input_channels", 1) or 1
        for ch in (req_ch, 1):
            use_ch = min(ch, max_in)
            if use_ch < 1:
                continue
            for rate in WANTED_RATES:
                try:
                    sd.check_input_settings(device=idx, samplerate=rate, channels=use_ch, dtype='int16')
                    return {"device": idx, "name": d["name"], "channels": use_ch,
                            "samplerate": rate, "max_in": max_in}
                except Exception:
                    continue
    raise RuntimeError("入力ありのデバイスが見つからない/希望設定で開けない")

@app.get("/health")
def health():
    result = {"llama": "down", "voicevox": "down", "ok": False}

    # LLM 健康チェック（軽いGETで十分）
    try:
        r = requests.get(f"{LLAMA_BASE}/v1/models", timeout=TIMEOUT)
        r.raise_for_status()
        result["llama"] = "up"
    except Exception as e:
        result["llama_error"] = str(e)[:160]

    # VOICEVOX 健康チェック（/speakers が軽くて安定）
    try:
        r = requests.get(f"{VOICEVOX_BASE}/speakers", timeout=TIMEOUT)
        r.raise_for_status()
        result["voicevox"] = "up"
    except Exception as e:
        result["voicevox_error"] = str(e)[:160]

    result["ok"] = (result["llama"] == "up" and result["voicevox"] == "up")
    return result

def _run(cmd: str, timeout: int = 60):
    """小さなヘルパー：subprocessを安全に回す"""
    try:
        completed = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
            text=True,
        )
        return completed.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"cmd failed: {cmd}\n{e.stderr.strip()[:300]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"cmd timeout: {cmd}")

@app.get("/audio/devices")
def list_audio_devices():
    devs = sd.query_devices()
    items = []
    for i, d in enumerate(devs):
        items.append({
            "index": i,
            "name": d["name"],
            "max_input": d.get("max_input_channels", 0),
            "max_output": d.get("max_output_channels", 0),
        })
    return {"devices": items, "prefer_order": PREFER_INPUT, "wanted_rates": WANTED_RATES}

@app.post("/listen")
def listen_and_transcribe():
    """
    1) マイクからREC_SECONDS秒録音（/tmp/in.wav）
    2) Whisperで文字起こしして（/tmp/out.txt）
    3) 文字を /transcripts にPOST → 音声で返答
    """
    wav = "/tmp/in.wav"
    txt = "/tmp/out.txt"

    # 1) 録音
    # Pulse/pipewire:
    rec_cmd = f'ffmpeg -y -f pulse -i default -t {REC_SECONDS} -ac 1 -ar 16000 -f wav {wav}'
    _run(rec_cmd, timeout=REC_SECONDS + 10)

    # 2) Whisper 実行
    whisper_cmd = os.getenv("WHISPER_CMD") 
    if not whisper_cmd:
        raise HTTPException(status_code=500, detail="WHISPER_CMD が未設定です（.env に書くか、コード内でwhisper_cmdを指定してね）")

    # `WHISPER_CMD`内の {wav} や {txtbase} を差し替えできるようにしておく
    whisper_cmd = whisper_cmd.format(wav=wav, txtbase="/tmp/out")
    _run(whisper_cmd, timeout=300)

    # openai-whisper は /tmp/in.txt、whisper.cppは /tmp/out.txt など、生成先が違うことがあるので両対応
    candidates = [txt, "/tmp/in.txt", "/tmp/out.txt"]
    text_path = next((p for p in candidates if os.path.exists(p)), None)
    if not text_path:
        raise HTTPException(status_code=500, detail="文字起こしファイルが見つからないよ（/tmp/out.txt 等）")

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        transcript = f.read().strip()

    if not transcript:
        raise HTTPException(status_code=400, detail="音声からテキストが取れなかったみたい")

    # 3) 既存の /transcripts に渡して返答フローを再利用
    r = requests.post("http://localhost:8000/transcripts", json={"text": transcript}, timeout=120)
    r.raise_for_status()
    return {"heard": transcript, "reply": r.json().get("reply")}

def write_wav(path, pcm16):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(VAD_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

def mix_to_mono_int16(arr: np.ndarray) -> np.ndarray:
    # 形状: (N,), (N,1), (N,2+) どれでもOKにする
    if arr.ndim == 1:
        return arr  # 既にmono
    if arr.shape[1] == 1:
        return arr[:, 0]  # (N,1) → (N,)

    # 2ch以上
    if CHANNEL_STRATEGY == "left":
        return arr[:, 0]
    if CHANNEL_STRATEGY == "right":
        return arr[:, 1]
    if CHANNEL_STRATEGY == "mean":
        return (arr.astype(np.int32).mean(axis=1)).astype(np.int16)

    # 既定: “最大振幅のch” をサンプル毎に選択（符号は保持）
    idx = np.abs(arr.astype(np.int32)).argmax(axis=1)
    return arr[np.arange(arr.shape[0]), idx]

def run_vad_loop():
    # ---- デバイス & 設定決定 ----
    req_ch = int(os.getenv("INPUT_CHANNELS", "2"))
    dev_sel = os.getenv("INPUT_DEVICE")  # 数字 or 名前 or 未設定
    picked = None

    if dev_sel:
        try:
            # 数字ならそのまま、名前なら部分一致で検索
            try:
                idx = int(dev_sel)
                info = sd.query_devices(idx, kind='input')
            except ValueError:
                all_devs = sd.query_devices()
                idx = next(i for i, d in enumerate(all_devs)
                           if (d.get("max_input_channels", 0) or 0) >= 1 and dev_sel in d["name"])
                info = sd.query_devices(idx, kind='input')

            max_in = info.get("max_input_channels", 0) or 0
            if max_in < 1:
                raise RuntimeError("指定デバイスは入力0")

            ok = None
            for ch in (req_ch, 1):
                use_ch = min(ch, max_in)
                if use_ch < 1:
                    continue
                for rate in WANTED_RATES:
                    try:
                        sd.check_input_settings(device=idx, samplerate=rate, channels=use_ch, dtype='int16')
                        ok = {"device": idx, "name": info["name"], "channels": use_ch,
                              "samplerate": rate, "max_in": max_in}
                        break
                    except Exception:
                        pass
                if ok:
                    break
            picked = ok or _pick_input_device(req_ch)
        except Exception:
            picked = _pick_input_device(req_ch)
    else:
        picked = _pick_input_device(req_ch)


    global VAD_SAMPLE_RATE
    VAD_SAMPLE_RATE = picked["samplerate"]         # ← WAVのframerateと一致させるため更新
    vad = webrtcvad.Vad(VAD_SENSITIVITY)

    frame_len = int(VAD_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
    silence_limit = int(SILENCE_TAIL_MS / VAD_FRAME_MS)

    buffer = []
    speaking = False
    silent_count = 0

    def callback(indata, frames, time_info, status):
        nonlocal speaking, silent_count, buffer
        if status:
            logging.warning(f"[vad] status={status}")

        # dtype=int16 で直接受け取る → そのままVADに渡せる
        data = indata  # int16
        pcm16 = mix_to_mono_int16(data)
        # フレーム境界に揃える（blocksize==frame_lenにしてるのでほぼ常に1回転でOK）
        for i in range(0, len(pcm16), frame_len):
            chunk = pcm16[i:i+frame_len]
            if len(chunk) < frame_len:
                break
            is_speech = vad.is_speech(chunk.tobytes(), sample_rate=VAD_SAMPLE_RATE)

            if is_speech and not speaking:
                logging.info("[vad] ▶ start speech")
            if (not is_speech) and speaking and silent_count+1 >= silence_limit:
                logging.info("[vad] ■ end speech (segment commit)")

            if is_speech:
                speaking = True
                silent_count = 0
                buffer.append(chunk.copy())
            else:
                if speaking:
                    silent_count += 1
                    if silent_count >= silence_limit:
                        # セグメント確定
                        speaking = False
                        audio = np.concatenate(buffer) if buffer else None
                        buffer = []
                        if audio is not None:
                            dur_ms = int(len(audio) * 1000 / VAD_SAMPLE_RATE)
                            seg_db = dbfs(audio)
                            if dur_ms >= 600 and seg_db > -45:  # 例の簡易ゲート（任意）
                                try:
                                    AUDIO_Q.put_nowait(audio.copy())   # ★ ここ大事：copy() で内容固定
                                    logging.info(f"[vad] enqueued segment len={dur_ms}ms, q={AUDIO_Q.qsize()}, level={seg_db:.1f} dBFS")
                                except queue.Full:
                                    logging.warning("[vad] queue full: dropping segment")
        return None, sd.CallbackFlags()

    # ★ ここが重要：dtype=int16 / blocksize=frame_len / device指定オプション
    stream_kwargs = dict(
        device=picked["device"],
        channels=picked["channels"],
        samplerate=VAD_SAMPLE_RATE,
        dtype='int16',
        blocksize=frame_len,
        callback=callback,
    )
    logging.info(f"[vad] using device={picked['device']} name='{picked['name']}', "
                 f"max_in={picked['max_in']} → channels={picked['channels']} rate={picked['samplerate']}")
    with sd.InputStream(**stream_kwargs):
        while LISTEN_ENABLED:
            time.sleep(0.1)

# FastAPI起動時にバックグラウンドで開始
@app.on_event("startup")
def _start_vad_if_enabled():
    global WORKER_THREAD, VAD_THREAD
    if LISTEN_ENABLED:
        WORKER_THREAD = threading.Thread(target=vad_worker, daemon=True)
        WORKER_THREAD.start()
        VAD_THREAD = threading.Thread(target=run_vad_loop, daemon=True)
        VAD_THREAD.start()

@app.on_event("shutdown")
def _graceful_stop():
    # 1) VAD 監視ループを止める
    global LISTEN_ENABLED
    LISTEN_ENABLED = False
    # 2) ワーカーへ終了合図（None を入れる）
    try:
        AUDIO_Q.put_nowait(None)
    except queue.Full:
        # いっぱいなら一度ブロックしてでも入れる
        AUDIO_Q.put(None)

class Transcript(BaseModel):
    text: str
@app.post("/transcripts")
def handle_transcript(t: Transcript):
    # 1) Llama.cppに投げる（OpenAI互換）
    payload = {
        "model": "local-model", # llama.cpp側のデフォ名で通ることが多い（未指定でも可な実装も）
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": t.text}
        ],
        "temperature": 0.9,
        "max_tokens": 256
    }
    r = requests.post(f"{LLAMA_BASE}/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    reply_text = r.json()["choices"][0]["message"]["content"]

    # 2) VOICEVOXで合成
    speaker_id = 47  # 好きな話者IDに変更
    q = requests.post(f"{VOICEVOX_BASE}/audio_query",
                      params={"text": reply_text, "speaker": speaker_id}, timeout=30)
    q.raise_for_status()
    s = requests.post(f"{VOICEVOX_BASE}/synthesis",
                      params={"speaker": speaker_id}, json=q.json(), timeout=60)
    s.raise_for_status()

    # 3) wav保存＆再生
    out_path = "/tmp/reply.wav"
    with open(out_path, "wb") as f:
        f.write(s.content)

    # 再生（PulseAudioなら paplay, ALSAなら aplay）
    try:
        subprocess.Popen(["paplay", out_path])
    except FileNotFoundError:
        subprocess.Popen(["aplay", out_path])

    return {"reply": reply_text}