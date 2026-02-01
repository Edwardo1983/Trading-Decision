from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def play_sound(path: str) -> None:
    try:
        from playsound import playsound
    except Exception as exc:
        logger.debug("playsound not available: %s", exc)
        return

    audio_path = Path(path)
    if not audio_path.exists():
        logger.debug("Sound file not found: %s", audio_path)
        return

    try:
        playsound(str(audio_path))
    except Exception as exc:
        logger.debug("Sound playback failed: %s", exc)