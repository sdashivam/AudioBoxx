from __future__ import annotations

import os
import sys, time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from TTS.api import TTS

SRC_DIR = Path(__file__).resolve().parents[1]
SEED_VC_DIR = Path(__file__).resolve().parent / "seed-vc"

sys.path.insert(0, str(SRC_DIR))
if SEED_VC_DIR.exists():
    sys.path.insert(0, str(SEED_VC_DIR))

from config import load_config

try:
    from seed_vc_wrapper import SeedVCWrapper

    SEED_VC_AVAILABLE = True
    SEED_VC_IMPORT_ERROR = ""
except ImportError as exc:
    SeedVCWrapper = None
    SEED_VC_AVAILABLE = False
    SEED_VC_IMPORT_ERROR = str(exc)
except Exception as exc:
    SeedVCWrapper = None
    SEED_VC_AVAILABLE = False
    SEED_VC_IMPORT_ERROR = str(exc)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Run Seed-VC setup from its repo so its relative cache paths stay stable."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@lru_cache(maxsize=2)
def get_tts_model(model_name: str, device: str) -> TTS:
    return TTS(model_name).to(device)


@lru_cache(maxsize=1)
def get_seed_vc_model(device: str) -> Any:
    if not SEED_VC_AVAILABLE or SeedVCWrapper is None:
        return None

    # Seed-VC uses huggingface_hub downloads with a local cache. Once the files
    # exist, later initializations reuse the cache instead of downloading again.
    with working_directory(SEED_VC_DIR):
        return SeedVCWrapper(device=torch.device(device))


class AudioGenerator:
    """
    Generate final translated audio by converting, timing, and merging each segment.
    """

    def __init__(self) -> None:
        self.config = load_config()
        self.device = self._resolve_device(self.config["tts"].get("device"))

        self.tts_model = self.config["tts"]["model_name"]
        self.tts_language = self.config["tts"]["language"]
        self.output_path = Path(self.config["paths"]["output_audio"])
        self.output_format = self.config["tts"]["output_format"].lower()
        self.speaker_reference = self.config["tts"]["speaker_reference"] or self.config["paths"]["input_audio"]

        self.temp_dir = Path(self.config["paths"]["temp_dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg_path = self.config["paths"].get("ffmpeg_path")
        if ffmpeg_path:
            os.environ["PATH"] += os.pathsep + ffmpeg_path
            ffmpeg_exe = Path(ffmpeg_path) / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                AudioSegment.converter = str(ffmpeg_exe)

        self.seed_vc_config = self.config.get("seed_vc", {})
        self.use_seed_vc = bool(self.seed_vc_config.get("enabled", True))
        self.target_voice_path = self.seed_vc_config.get("target_voice_path") or self.config["paths"]["input_audio"]

        self.tts = get_tts_model(self.tts_model, self.device)
        self.seed_vc = get_seed_vc_model(self.device) if self.use_seed_vc else None

        if self.use_seed_vc and self.seed_vc is None:
            print(f"Seed-VC is not available: {SEED_VC_IMPORT_ERROR}")
            print("Falling back to simple pitch/tone matching.")

    def _resolve_device(self, configured_device: str | None) -> str:
        if configured_device:
            if configured_device == "cuda" and not torch.cuda.is_available():
                print("CUDA requested for audio generation, but CUDA is not available. Using CPU.")
                return "cpu"
            return configured_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate_audio_from_segments(
                self,
                translated_segments: list[dict[str, Any]],
                speaker_wav: str | None = None,
            ) -> dict:

        print("Starting audio generation pipeline...")

        metrics = {}

        # ---- Tone Matching (XTTS / Seed-VC) ---- #
        t0 = time.time()
        tone_matched_files = self.match_segment_tone(
            translated_segments, speaker_wav=speaker_wav
        )
        metrics["tone_time"] = time.time() - t0

        # ---- Alignment ---- #
        t1 = time.time()
        aligned_files = self.align_segment_durations(
            tone_matched_files, translated_segments
        )
        metrics["align_time"] = time.time() - t1

        # ---- Merge ---- #
        t2 = time.time()
        output_audio = self.merge_audio_segments(
            aligned_files, translated_segments
        )
        metrics["merge_time"] = time.time() - t2

        print(f"Audio generation complete. Output: {output_audio}")

        return {
            "output_path": output_audio,
            "aligned_files": aligned_files,
            "metrics": metrics
        }

    def match_segment_tone(
        self,
        segments: list[dict[str, Any]],
        speaker_wav: str | None = None,
    ) -> list[str]:
        audio_files = []
        speaker_ref = speaker_wav or self.speaker_reference

        for index, segment in enumerate(segments):
            source_audio = self._get_or_create_segment_tts(segment, index, speaker_ref)
            matched_audio = self.temp_dir / f"segment_{index}_tone_matched.wav"

            print(f"Matching tone for segment {index}...")
            converted = self.apply_voice_conversion(source_audio, matched_audio)
            audio_files.append(str(converted))

        return audio_files

    def _get_or_create_segment_tts(
        self,
        segment: dict[str, Any],
        index: int,
        speaker_wav: str,
    ) -> str:
        existing_audio = segment.get("tts_audio")
        if existing_audio and Path(existing_audio).exists():
            return str(existing_audio)

        text = segment.get("translated_text") or segment.get("text") or ""
        if not text.strip():
            return self._create_silence(index, segment)

        output_path = self.temp_dir / f"segment_{index}_tts.wav"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=self.tts_language,
            file_path=str(output_path),
        )
        return str(output_path)

    def apply_voice_conversion(self, audio_path: str | Path, output_path: str | Path) -> str:
        if self.seed_vc is not None:
            try:
                return self._apply_seed_vc_conversion(audio_path, output_path)
            except Exception as exc:
                print(f"Seed-VC failed for {audio_path}: {exc}. Using simple tone matching.")

        return self._apply_simple_tone_matching(audio_path, output_path)

    def _apply_seed_vc_conversion(self, audio_path: str | Path, output_path: str | Path) -> str:
        output_path = Path(output_path)

        with working_directory(SEED_VC_DIR):
            generated = self.seed_vc.convert_voice(
                source=str(Path(audio_path).resolve()),
                target=str(Path(self.target_voice_path).resolve()),
                diffusion_steps=int(self.seed_vc_config.get("diffusion_steps", 10)),
                length_adjust=float(self.seed_vc_config.get("length_adjust", 1.0)),
                inference_cfg_rate=float(self.seed_vc_config.get("inference_cfg_rate", 0.7)),
                f0_condition=bool(self.seed_vc_config.get("f0_condition", False)),
                auto_f0_adjust=bool(self.seed_vc_config.get("auto_f0_adjust", True)),
                pitch_shift=float(self.seed_vc_config.get("pitch_shift", 0)),
                stream_output=False,
            )

        audio, sample_rate = self._normalize_seed_vc_output(generated)
        sf.write(output_path, audio, sample_rate)
        return str(output_path)

    def _normalize_seed_vc_output(self, generated: Any) -> tuple[np.ndarray, int]:
        sample_rate = 22050
        audio = generated

        if isinstance(generated, tuple) and len(generated) == 2:
            sample_rate, audio = generated

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        return audio, int(sample_rate)

    def _apply_simple_tone_matching(self, audio_path: str | Path, output_path: str | Path) -> str:
        try:
            source_audio, sample_rate = librosa.load(audio_path, sr=None)
            target_audio, _ = librosa.load(self.target_voice_path, sr=sample_rate)
            target_pitch = self._mean_pitch(target_audio, sample_rate)
            source_pitch = self._mean_pitch(source_audio, sample_rate)

            if source_pitch > 0 and target_pitch > 0:
                pitch_ratio = target_pitch / source_pitch
                pitch_steps = float(np.clip(np.log2(pitch_ratio) * 12, -8, 8))
                source_audio = librosa.effects.pitch_shift(source_audio, sr=sample_rate, n_steps=pitch_steps)

            sf.write(output_path, source_audio, sample_rate)
            return str(output_path)
        except Exception as exc:
            print(f"Simple tone matching failed for {audio_path}: {exc}. Using original segment audio.")
            return str(audio_path)

    def _mean_pitch(self, audio: np.ndarray, sample_rate: int) -> float:
        pitches, _ = librosa.piptrack(y=audio, sr=sample_rate)
        voiced = pitches[pitches > 0]
        return float(np.mean(voiced)) if voiced.size else 0.0

    def align_segment_durations(
        self,
        audio_files: list[str],
        segments: list[dict[str, Any]],
    ) -> list[str]:
        aligned_files = []

        for index, (audio_file, segment) in enumerate(zip(audio_files, segments)):
            target_duration = max(float(segment["end"]) - float(segment["start"]), 0.05)
            output_path = self.temp_dir / f"segment_{index}_aligned.wav"
            self._align_single_duration(audio_file, target_duration, output_path)
            aligned_files.append(str(output_path))
            print(f"Aligned segment {index} to {target_duration:.2f}s")

        return aligned_files

    def _align_single_duration(
        self,
        input_path: str | Path,
        target_duration: float,
        output_path: str | Path,
    ) -> None:
        audio, sample_rate = librosa.load(input_path, sr=None)
        current_duration = len(audio) / sample_rate if sample_rate else 0

        if current_duration <= 0:
            self._write_silence(output_path, target_duration)
            return

        rate = current_duration / target_duration
        rate = float(np.clip(rate, 0.25, 4.0))
        aligned_audio = librosa.effects.time_stretch(audio, rate=rate)

        target_samples = max(int(target_duration * sample_rate), 1)
        if len(aligned_audio) > target_samples:
            aligned_audio = aligned_audio[:target_samples]
        elif len(aligned_audio) < target_samples:
            aligned_audio = np.pad(aligned_audio, (0, target_samples - len(aligned_audio)))

        sf.write(output_path, aligned_audio, sample_rate)

    def merge_audio_segments(self, audio_files: list[str], segments: list[dict[str, Any]]) -> str:
        if not audio_files:
            raise ValueError("No audio files were generated for merging.")

        final_duration_ms = int(max(float(segment["end"]) for segment in segments) * 1000)
        final_audio = AudioSegment.silent(duration=max(final_duration_ms, 1))

        for index, (audio_file, segment) in enumerate(zip(audio_files, segments)):
            segment_audio = AudioSegment.from_file(audio_file)
            start_ms = max(int(float(segment["start"]) * 1000), 0)
            final_audio = final_audio.overlay(segment_audio, position=start_ms)
            print(f"Merged segment {index} at {start_ms}ms")

        final_audio = self._normalize_audio(final_audio)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "mp3":
            final_audio.export(self.output_path, format="mp3", bitrate="192k")
        else:
            final_audio.export(self.output_path, format="wav")

        return str(self.output_path)

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        if audio.dBFS == float("-inf"):
            return audio

        gain = min(-1.0 - audio.dBFS, 6.0)
        return audio.apply_gain(gain) if gain > 0 else audio

    def _create_silence(self, index: int, segment: dict[str, Any]) -> str:
        duration = max(float(segment["end"]) - float(segment["start"]), 0.05)
        output_path = self.temp_dir / f"segment_{index}_silence.wav"
        self._write_silence(output_path, duration)
        return str(output_path)

    def _write_silence(self, output_path: str | Path, duration: float, sample_rate: int = 22050) -> None:
        silence = np.zeros(max(int(duration * sample_rate), 1), dtype=np.float32)
        sf.write(output_path, silence, sample_rate)
