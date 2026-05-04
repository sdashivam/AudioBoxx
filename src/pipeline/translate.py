import argostranslate.package
import argostranslate.translate
import os
import sys
from pathlib import Path
from typing import Any

import torch
from TTS.api import TTS

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import load_config


class Translator:
    def __init__(self):
        """Initialize translator and TTS model with config parameters"""
        self.config = load_config()
        self.from_lang = self.config["translation"]["from_lang"]
        self.to_lang = self.config["translation"]["to_lang"]
        self.tts_model = self.config["tts"]["model_name"]
        self.tts_language = self.config["tts"]["language"]
        self.speaker_reference = self.config["tts"]["speaker_reference"] or self.config["paths"]["input_audio"]
        self.temp_dir = Path(self.config["paths"]["temp_dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.device = self.config["tts"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_language_packages()
        self.tts = TTS(self.tts_model).to(self.device)

    def _setup_language_packages(self) -> None:
        """Install language packages if not already installed"""
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            
            pkg = next(
                (p for p in available_packages 
                 if p.from_code == self.from_lang and p.to_code == self.to_lang),
                None
            )
            
            if pkg:
                argostranslate.package.install_from_path(pkg.download())
                print(f"Installed {self.from_lang} -> {self.to_lang} language package")
        except Exception as e:
            print(f"Warning: Could not setup language packages: {e}")

    def translate_text(self, text: str) -> str:
        """Translate a single text string"""
        return argostranslate.translate.translate(text, self.from_lang, self.to_lang)

    def text_to_speech(self, text: str, output_path: str | Path, speaker_wav: str | None = None) -> str:
        """Generate TTS audio for translated text and return the output file path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav or self.speaker_reference,
            language=self.tts_language,
            file_path=str(output_path),
        )

        return str(output_path)

    def translate_segments(
        self,
        segments: list[dict[str, Any]],
        generate_tts: bool = True,
        speaker_wav: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Translate text in each segment and optionally generate TTS audio per segment.
        
        Args:
            segments: List of segment dicts from AudioSegmenter
                     Expected format: {"start": float, "end": float, "text": str, ...}
            generate_tts: Whether to synthesize translated text into speech.
            speaker_wav: Optional speaker reference audio for multilingual TTS models.
        
        Returns:
            List of translated segments with timing, translated text, and TTS audio path.
        """
        translated_segments = []
        
        for index, seg in enumerate(segments):
            original_text = seg.get("text", "")
            translated_text = self.translate_text(original_text) if original_text.strip() else ""

            translated_seg = seg.copy()
            translated_seg["original_text"] = original_text
            translated_seg["text"] = translated_text
            translated_seg["translated_text"] = translated_text

            if generate_tts and translated_text.strip():
                tts_output_path = self.temp_dir / f"translated_segment_{index}.wav"
                translated_seg["tts_audio"] = self.text_to_speech(
                    translated_text,
                    tts_output_path,
                    speaker_wav=speaker_wav,
                )

            translated_segments.append(translated_seg)
        
        return translated_segments
