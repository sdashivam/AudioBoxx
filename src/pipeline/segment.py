import os
import torch
import whisper
from pathlib import Path
from typing import Any
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import load_config


class AudioSegmenter:
    def __init__(
        self,
        model_name: str = None,
        device: str | None = None,
        ffmpeg_path: str | None = None,
    ) -> None:
        # Load config and use defaults from config if not provided
        self.config = load_config()

        self.model_name = model_name or self.config["whisper"]["model_name"]
        self.device = device or self.config["whisper"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ffmpeg_path = ffmpeg_path or self.config["paths"]["ffmpeg_path"]

        if self.ffmpeg_path:
            os.environ["PATH"] += os.pathsep + self.ffmpeg_path

        self.model = whisper.load_model(self.model_name).to(self.device)

    def transcribe(self, audio_file: str | Path, **options: Any) -> dict[str, Any]:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use config defaults for transcribe options
        transcribe_options = {
            "language": self.config["whisper"]["language"],
            "beam_size": self.config["whisper"]["beam_size"],
            "best_of": self.config["whisper"]["best_of"],
            "temperature": self.config["whisper"]["temperature"],
            "condition_on_previous_text": self.config["whisper"]["condition_on_previous_text"],
        }
        transcribe_options.update(options)

        return self.model.transcribe(str(audio_path), **transcribe_options)

    def get_segments(self, audio_file: str | Path, **options: Any) -> list[dict[str, Any]]:
        result = self.transcribe(audio_file, **options)
        return result["segments"]
    
