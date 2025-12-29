import logging
from pathlib import Path


def setup_logging(log_path: str = "./app.log", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s : [%(name)s] %(message)s - (%(filename)s : %(lineno)s)"
    )

    # コンソール出力用
    s_handler = logging.StreamHandler()
    s_handler.setFormatter(log_formatter)
    s_handler.setLevel(level)
    root.addHandler(s_handler)

    # ログファイル保存用
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f_handler = logging.FileHandler(log_path, encoding="utf-8")
    f_handler.setFormatter(log_formatter)
    f_handler.setLevel(logging.DEBUG)
    root.addHandler(f_handler)
