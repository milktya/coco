import sqlite3
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional

JST = ZoneInfo("Asia/Tokyo")


def now_jst_iso() -> str:  # 例: "2026-01-02T23:12:34+09:00"
    return datetime.now(JST).isoformat()


def init_db(db_path: str = "data/coco.db") -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
            content TEXT NOT NULL,
            created_at_jst TEXT NOT NULL
        );
        """
        )
        # 直近取得を速くする
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_id ON messages(id);")
        conn.commit()
    finally:
        conn.close()


def save_message(role: str, content: str, db_path: str = "data/coco.db") -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (role, content, created_at_jst) VALUES (?, ?, ?)",
            (role, content, now_jst_iso()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def load_recent_messages(
    limit: int = 10, db_path: str = "data/coco.db"
) -> List[Tuple[str, str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, created_at_jst FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        rows.reverse()  # 返り値: [(role, content), ...] で古い→新しい順に並べる
        return rows
    finally:
        conn.close()
