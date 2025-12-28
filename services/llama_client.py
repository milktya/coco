import requests
import config


def chat(transcript: str) -> str:
    # Llama.cppに投げる内容（OpenAI互換）
    payload = {
        "model": "local-model",  # 厳密なモデル名を入れなくてもllama.cpp側のデフォ名でOK
        "messages": [
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            },
            {"role": "user", "content": transcript},
        ],
        # モデル設定
        "temperature": 0.9,
        "max_tokens": 256,
    }
    # 内容をJSONにして送信
    r = requests.post(f"{config.LLAMA_BASE}/chat/completions", json=payload, timeout=60)
    # 返答を受信
    r.raise_for_status()
    reply_text = r.json()["choices"][0]["message"]["content"]

    return reply_text
