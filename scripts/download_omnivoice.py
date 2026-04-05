#!/usr/bin/env python3
"""Скачать веса k2-fsa/OmniVoice в локальную папку (без загрузки при старте приложения).

По умолчанию каталог: ../omnivoice-model/ относительно scripts/ (корень репозитория), т.е. рядом с папкой scripts.

Примеры:
  python3 scripts/download_omnivoice.py
  python3 scripts/download_omnivoice.py --dir /data/OmniVoice
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/download_omnivoice.py

Дальше в Docker: смонтируйте эту папку в /model и задайте OMNIVOICE_MODEL=/model
(см. комментарии в docker-compose.yml).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_local_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "omnivoice-model"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo",
        default="k2-fsa/OmniVoice",
        help="Репозиторий на Hugging Face Hub",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Куда сохранить (по умолчанию: {default_local_dir()})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Токен Hugging Face (или переменная окружения HF_TOKEN)",
    )
    args = parser.parse_args()
    out: Path = args.dir.expanduser().resolve() if args.dir else default_local_dir()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Установите: pip install huggingface_hub")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    print(f"Репозиторий: {args.repo}")
    print(f"Каталог:     {out}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        token=args.token,
    )
    print("Готово. Укажите этот путь в OMNIVOICE_MODEL (в контейнере — путь монтирования, напр. /model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
