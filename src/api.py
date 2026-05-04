import io
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import load_config
from pipeline.generation import AudioGenerator
from pipeline.segment import AudioSegmenter
from pipeline.translate import Translator


app = FastAPI(
    title="AudioBoxx API",
    description="Audio Dubbing Pipeline.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_pipeline() -> "AudioBoxxPipeline":
    return AudioBoxxPipeline()


class AudioBoxxPipeline:
    def __init__(self) -> None:
        self.config = load_config()
        self.segmenter = AudioSegmenter()
        self.translator = Translator()
        self.generator = AudioGenerator()

    def segment(self, audio_path: str | Path) -> list[dict[str, Any]]:
        return self.segmenter.get_segments(audio_path)

    def translate(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.translator.translate_segments(segments)

    def process(self, audio_path: str | Path) -> dict[str, Any]:
        logs = io.StringIO()
        audio_path = Path(audio_path)

        with redirect_stdout(logs):
            print("Step 1: Segmenting audio...")
            segments = self.segment(audio_path)
            print(f"Found {len(segments)} segments")

            print("\nStep 2: Translating segments...")
            translated_segments = self.translate(segments)
            print(f"Translated {len(translated_segments)} segments")

            print("\nStep 3: Generating final audio...")
            output_audio = self.generator.generate_audio_from_segments(translated_segments)
            print(f"Pipeline complete. Final audio saved to: {output_audio}")

        return {
            "input_audio": str(audio_path),
            "output_audio": output_audio,
            "from_lang": self.config["translation"]["from_lang"],
            "to_lang": self.config["translation"]["to_lang"],
            "segment_count": len(segments),
            "segments": serialize_segments(segments),
            "translated_segments": serialize_segments(translated_segments),
            "logs": logs.getvalue(),
        }


def save_upload(uploaded_file: UploadFile, destination: str) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        file.write(uploaded_file.file.read())

    return output_path


def serialize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for segment in segments:
        serialized.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", ""),
            }
        )
    return serialized


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "AudioBoxx API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    config = load_config()
    return {
        "paths": config["paths"],
        "whisper": config["whisper"],
        "translation": config["translation"],
        "tts": config["tts"],
    }


@app.post("/process")
def process_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    config = load_config()
    audio_path = save_upload(file, config["paths"]["input_audio"])

    result = get_pipeline().process(audio_path)
    output_audio = result.get("output_audio")

    if not output_audio or not Path(output_audio).exists():
        raise HTTPException(status_code=500, detail="Pipeline finished without creating output audio.")

    result["download_url"] = "/download"
    return result


@app.get("/download")
def download_output() -> FileResponse:
    config = load_config()
    output_audio = Path(config["paths"]["output_audio"])

    if not output_audio.exists():
        raise HTTPException(status_code=404, detail="Output audio file not found.")

    media_type = "audio/wav"
    if output_audio.suffix.lower() == ".mp3":
        media_type = "audio/mpeg"

    return FileResponse(
        path=str(output_audio),
        media_type=media_type,
        filename=output_audio.name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
