import sys, os, time, librosa
import uuid
import json
from datetime import datetime
import torch

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from config import load_config
from pipeline.segment import AudioSegmenter
from pipeline.translate import Translator
from pipeline.generation import AudioGenerator
from utils.logger import setup_logger

# ---------------- LOGGER ---------------- #
logger = setup_logger()

# ---------------- CONFIG ---------------- #
config = load_config()

# ---------------- COMPONENTS ---------------- #
segmenter = AudioSegmenter()
translator = Translator()
generator = AudioGenerator()

# ---------------- HELPERS ---------------- #
def save_metrics(metrics: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"\nMetrics saved to: {output_path}")


def compute_sync_error(aligned_files, segments):
    errors = []
    for i, file in enumerate(aligned_files):
        y, sr = librosa.load(file)
        gen_dur = len(y) / sr
        target = segments[i]["end"] - segments[i]["start"]
        errors.append(abs(gen_dur - target))
    return sum(errors) / len(errors)


# ---------------- API CONTEXT ---------------- #
request_id = str(uuid.uuid4())[:8]
api_start_time = time.time()
api_status = "success"

# ---------------- PIPELINE ---------------- #
try:
    logger.info("\nStarting pipeline...")

    # ---- Load audio ---- #
    audio, sr = librosa.load(config["paths"]["input_audio"])
    audio_duration = len(audio) / sr

    # ---- Step 1: Segmentation ---- #
    seg_start = time.time()
    logger.info("Step 1: Segmenting audio...")
    segments = segmenter.get_segments(config["paths"]["input_audio"])
    seg_time = time.time() - seg_start
    logger.info(f"Segments found: {len(segments)}")

    # ---- Step 2: Translation ---- #
    trans_start = time.time()
    logger.info("Step 2: Translating segments...")

    failures = []
    try:
        translated_segments = translator.translate_segments(segments)
    except Exception as e:
        failures.append({"stage": "translation", "error": str(e)})
        api_status = "failed"
        raise e

    trans_time = time.time() - trans_start
    success_rate = len(translated_segments) / len(segments)

    # ---- Step 3: Generation ---- #
    gen_start = time.time()
    logger.info("Step 3: Generating final audio...")

    result = generator.generate_audio_from_segments(translated_segments)

    final_audio_path = result["output_path"]
    aligned_files = result["aligned_files"]
    gen_metrics = result["metrics"]

    gen_time = time.time() - gen_start

    # ---- Pipeline end ---- #
    pipeline_end = time.time()
    total_time = pipeline_end - api_start_time
    rtf = total_time / audio_duration

    # ---- Sync Error ---- #
    sync_error = compute_sync_error(aligned_files, segments)

except Exception as e:
    logger.exception("\nPipeline failed")
    api_status = "failed"

    # fallback values (avoid crash)
    final_audio_path = None
    segments = []
    aligned_files = []
    gen_metrics = {}
    success_rate = 0
    sync_error = None
    total_time = time.time() - api_start_time
    audio_duration = 1  # avoid divide by zero
    rtf = 0

# ---------------- API METRICS ---------------- #
api_end_time = time.time()
api_latency = api_end_time - api_start_time

# ---------------- METRICS OBJECT ---------------- #
metrics = {
    "run_info": {
        "timestamp": datetime.now().isoformat(),
        "run_id": request_id,
        "input_audio": config["paths"]["input_audio"],
        "output_audio": final_audio_path
    },
    "audio": {
        "duration_sec": round(audio_duration, 2),
        "num_segments": len(segments)
    },
    "execution": {
        "total_time_sec": round(total_time, 2),
        "rtf": round(rtf, 2),
        "segmentation_time_sec": round(seg_time, 2) if 'seg_time' in locals() else None,
        "translation_time_sec": round(trans_time, 2) if 'trans_time' in locals() else None,
        "generation_time_sec": round(gen_time, 2) if 'gen_time' in locals() else None
    },
    "quality": {
        "sync_error_sec": round(sync_error, 3) if sync_error else None,
        "success_rate": round(success_rate, 3)
    },
    "generation_breakdown": {
        k: round(v, 2) for k, v in gen_metrics.items()
    },
    "failures": {
        "count": len(failures) if 'failures' in locals() else 0,
        "details": failures if 'failures' in locals() else []
    },
    "api": {
        "request_id": request_id,
        "status": api_status,
        "latency_sec": round(api_latency, 2),
        "rtf_per_request": round(api_latency / audio_duration, 2)
    }
}

# ---- GPU Metrics ---- #
if torch.cuda.is_available():
    metrics["execution"]["gpu_memory_mb"] = round(
        torch.cuda.max_memory_allocated() / 1e6, 2
    )

# ---------------- SAVE METRICS ---------------- #
metrics_file = f"metrics_{request_id}.json"
metrics_path = os.path.join(config['paths']['output_dir'], metrics_file)
save_metrics(metrics, metrics_path)

# ---------------- LOG SUMMARY ---------------- #
logger.info("\nSUMMARY")
logger.info(f"RTF: {rtf:.2f}")
logger.info(f"Sync Error: {sync_error}")
logger.info(f"Success Rate: {success_rate*100:.2f}%")
logger.info(f"Output: {final_audio_path}")

logger.info("\nAPI METRICS")
logger.info(f"Request ID: {request_id}")
logger.info(f"Status: {api_status}")
logger.info(f"Latency: {api_latency:.2f}s")
logger.info(f"RTF (API): {api_latency / audio_duration:.2f}")

logger.info("\nPipeline completed.")