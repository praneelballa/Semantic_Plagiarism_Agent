import os

os.environ["PATH"] += os.pathsep + r"C:\Users\acsma\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin"
import shutil
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Union
import whisper

from app.config import settings
from app.core.logging import logger
from app.core.normalizer import normalize_text


def _ensure_ffmpeg_on_path() -> None:
    """
    Guarantees that ffmpeg is discoverable by subprocesses on Windows.
    Scans common installation paths and appends to os.environ['PATH'].
    """
    if shutil.which("ffmpeg"):
        return

    # Typical directories where winget, choco, or manual setups install ffmpeg
    local_app_data = os.getenv("LOCALAPPDATA", "")
    program_data = os.getenv("ProgramData", "")
    user_profile = os.getenv("USERPROFILE", "")

    search_dirs = [
        r"C:\ffmpeg\bin",
        r"C:\ffmpeg",
        os.path.join(program_data, "chocolatey", "bin"),
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages"),
        os.path.join(user_profile, "AppData", "Local", "Microsoft", "WinGet", "Packages"),
    ]

    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                if "ffmpeg.exe" in files:
                    logger.info(f"Dynamically registered FFmpeg path: {root}")
                    os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
                    return


# Run path discovery at module import
_ensure_ffmpeg_on_path()


class WhisperTranscriber:
    """Local ASR engine powered by OpenAI Whisper."""

    def __init__(self, model_size: str = None):
        self.model_size = model_size or settings.WHISPER_MODEL
        logger.info(f"Loading local Whisper model: '{self.model_size}' on CPU...")
        self.model = whisper.load_model(self.model_size, device="cpu")
        logger.info("Whisper model loaded successfully.")

    def transcribe(
        self, audio_input: Union[str, Path, bytes], file_suffix: str = ".wav"
    ) -> List[Dict[str, Any]]:
        """
        Transcribes audio and returns timestamped segments:
        [
            {"text": "...", "start": 12.4, "end": 28.1}
        ]
        """
        temp_file_path = None
        try:
            if isinstance(audio_input, bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                    tmp.write(audio_input)
                    temp_file_path = tmp.name
                file_path = temp_file_path
            else:
                file_path = str(audio_input)

            result = self.model.transcribe(file_path, fp16=False)
            segments = result.get("segments", [])

            processed_segments = []
            for seg in segments:
                clean_text = normalize_text(seg.get("text", ""))
                if clean_text:
                    processed_segments.append({
                        "text": clean_text,
                        "start": round(float(seg.get("start", 0.0)), 2),
                        "end": round(float(seg.get("end", 0.0)), 2),
                    })

            return processed_segments

        finally:
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink(missing_ok=True)


@lru_cache(maxsize=1)
def get_transcriber() -> WhisperTranscriber:
    """Singleton cached provider for Whisper."""
    return WhisperTranscriber()