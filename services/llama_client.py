import logging
from typing import Any, Dict, List, Union

import requests
import config

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "...", "content": "..."}


# llama.cpp(OpenAI互換)に投げられる形に整形＆最低限バリデーション。
def _validate_messages(input_messages: List[Any]) -> List[Message]:
    messages: List[Message] = []
    for m in input_messages:
        if not isinstance(m, dict):
            raise TypeError("messages must be a list of dicts")
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("each message must have string 'role' and string 'content'")
        messages.append({"role": role, "content": content})
    return messages


# llama.cppへメッセージ送信
def chat(input_data: Union[str, List[Message]]) -> str:
    logger.info("llama.cpp ready...")
    # input_data が str のとき: transcript 1本の会話
    if isinstance(input_data, str):
        messages: List[Message] = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": input_data},
        ]
    # input_data が messages(list[dict]) のとき: そのまま会話履歴として投げる
    elif isinstance(input_data, list):
        messages = _validate_messages(input_data)
    else:
        raise TypeError("chat() input_data must be str or list of messages")

    payload: Dict[str, Any] = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 256,
    }

    r = requests.post(f"{config.LLAMA_BASE}/chat/completions", json=payload, timeout=60)
    r.raise_for_status()

    data = r.json()
    reply_text = data["choices"][0]["message"]["content"]
    logger.info(f"reply: {reply_text}")
    return reply_text
