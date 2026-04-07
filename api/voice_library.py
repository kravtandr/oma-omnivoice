"""Persistent, thread-safe library for reference voice audio files."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VoiceLibrary:
    """File-backed library of named reference voices.

    Layout::

        <base_dir>/
            index.json          ← name → metadata
            alice.wav
            bob_1.wav
            ...
    """

    def __init__(self, base_dir: str) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._lock = threading.Lock()
        self._index: Dict[str, Any] = {}
        self._load()
        logger.info("VoiceLibrary: %d голос(ов) загружено из %s", len(self._index), self._dir)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}

    def _load(self) -> None:
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    self._index = json.load(f)
                # Remove entries whose audio file has been deleted externally
                missing = [n for n, m in self._index.items() if not (self._dir / m["filename"]).exists()]
                for n in missing:
                    logger.warning("VoiceLibrary: файл %r удалён снаружи, удаляем из индекса", n)
                    del self._index[n]
                if missing:
                    self._save_unlocked()
            except Exception:
                logger.exception("VoiceLibrary: не удалось прочитать index, начинаем пустым")
                self._index = {}
        self._scan_untracked()

    def _scan_untracked(self) -> None:
        """Auto-import audio files found in the directory but not tracked in the index."""
        tracked_files = {m["filename"] for m in self._index.values()}
        new_entries = 0
        for p in sorted(self._dir.iterdir()):
            if p.suffix.lower() not in self._AUDIO_EXTENSIONS:
                continue
            if p.name in tracked_files:
                continue
            # Derive a human-readable name from the filename
            name = self._name_from_file(p.name)
            # Ensure name uniqueness
            if name in self._index:
                base = name
                counter = 1
                while name in self._index:
                    name = f"{base} ({counter})"
                    counter += 1
            meta: Dict[str, Any] = {
                "name": name,
                "filename": p.name,
                "ref_text": "",
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
                "size_bytes": p.stat().st_size,
            }
            self._index[name] = meta
            new_entries += 1
            logger.info("VoiceLibrary: автоимпорт %r ← %s", name, p.name)
        if new_entries:
            self._save_unlocked()

    @staticmethod
    def _name_from_file(filename: str) -> str:
        """Convert filename to a display name: strip prefix/extension, clean up."""
        name = Path(filename).stem  # drop extension
        # Strip common prefixes like "voice_preview_"
        for prefix in ("voice_preview_", "voice_", "ref_", "reference_"):
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break
        # Replace underscores/dashes with spaces and clean up
        name = name.replace("_", " ").strip()
        # Capitalise first letter
        if name:
            name = name[0].upper() + name[1:]
        return name or filename

    def _save_unlocked(self) -> None:
        """Write index to disk (call only while holding self._lock, or from _load)."""
        tmp = self._index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
        tmp.replace(self._index_path)

    def _save(self) -> None:
        self._save_unlocked()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_name(name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Имя голоса не должно быть пустым.")
        if len(name) > 120:
            raise ValueError("Имя голоса слишком длинное (максимум 120 символов).")
        if not re.match(r'^[\w\s\-\.]+$', name, re.UNICODE):
            raise ValueError(
                f"Недопустимое имя {name!r}: только буквы, цифры, пробелы, дефисы, точки, подчёркивания."
            )
        return name

    def _unique_filename(self, name: str) -> str:
        base = re.sub(r'[^\w\-]', '_', name.strip())[:80]
        candidate = base + ".wav"
        counter = 0
        while (self._dir / candidate).exists():
            counter += 1
            candidate = f"{base}_{counter}.wav"
        return candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        audio_bytes: bytes,
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save audio bytes under *name*. Raises ValueError if name already exists."""
        name = self.validate_name(name)
        with self._lock:
            if name in self._index:
                raise ValueError(f"Голос {name!r} уже существует. Удалите его сначала.")
            filename = self._unique_filename(name)
            (self._dir / filename).write_bytes(audio_bytes)
            meta: Dict[str, Any] = {
                "name": name,
                "filename": filename,
                "ref_text": (ref_text or "").strip(),
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size_bytes": len(audio_bytes),
            }
            self._index[name] = meta
            self._save()
        logger.info("VoiceLibrary: добавлен %r (%d байт)", name, len(audio_bytes))
        return dict(meta)

    def delete(self, name: str) -> bool:
        """Delete voice by name. Returns True if found and deleted."""
        with self._lock:
            if name not in self._index:
                return False
            meta = self._index.pop(name)
            audio_path = self._dir / meta["filename"]
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    logger.warning("VoiceLibrary: не удалось удалить файл %s", audio_path)
            self._save()
        logger.info("VoiceLibrary: удалён %r", name)
        return True

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for *name*, or None if not found."""
        with self._lock:
            m = self._index.get(name)
            return dict(m) if m else None

    def get_audio_path(self, name: str) -> Optional[str]:
        """Return absolute path to the audio file, or None if not found."""
        with self._lock:
            meta = self._index.get(name)
        if not meta:
            return None
        p = self._dir / meta["filename"]
        return str(p) if p.exists() else None

    def get_ref_text(self, name: str) -> Optional[str]:
        """Return stored reference transcript for *name*, or None."""
        meta = self.get(name)
        if not meta:
            return None
        t = meta.get("ref_text", "")
        return t if t else None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return list of all voice metadata dicts."""
        with self._lock:
            return [dict(v) for v in self._index.values()]

    def names(self) -> List[str]:
        """Return sorted list of voice names."""
        with self._lock:
            return sorted(self._index.keys())
