"""Audio features: 16kHz wav extraction, transcript, AI-voice probability.

Three stages, each independently swappable:

1. ``extract_wav`` — ffmpeg → mono 16kHz PCM. Uses the bundled ffmpeg already
   on PATH from ingest.
2. ``Transcriber`` — faster-whisper (small) → text + detected language.
3. ``VoiceDetector`` — wav2vec2 anti-spoof classifier → P(AI-generated voice).

Anti-spoof models are typically trained on short (~4s) windows. We chunk the
clip at 4s, score each chunk, and average — more robust than a single 30s pass
that the model would just truncate.

Models:
    faster-whisper "small" (CTranslate2 reimplementation of OpenAI Whisper).
        - lib:     https://github.com/SYSTRAN/faster-whisper
        - weights: https://huggingface.co/Systran/faster-whisper-small
          (auto-resolved from the "small" alias; ~244M params, multilingual.)

    Deepfake-audio-detection (wav2vec2 fine-tuned for real-vs-fake speech).
        - card:   https://huggingface.co/motheecreator/Deepfake-audio-detection
        - base:   facebook/wav2vec2-base
          PLAN.md flags AI-voice detector quality as varying by training set —
          spot-check before relying on this score (M9 error analysis).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

from medea.ingest.youtube import _ensure_ffmpeg_on_path

log = logging.getLogger(__name__)

WHISPER_MODEL = "small"
AI_VOICE_MODEL = "motheecreator/Deepfake-audio-detection"
TARGET_SR = 16_000
VOICE_CHUNK_SECONDS = 4
_REAL_HINTS = ("real", "bona", "human", "natural", "genuine")


@dataclass
class AudioFeatures:
    transcript: str
    language: str | None
    ai_voice_prob: float


def extract_wav(clip_path: Path, out_path: Path, sr: int = TARGET_SR) -> Path:
    """ffmpeg-decode the clip's audio track to mono PCM16 at ``sr`` Hz."""
    _ensure_ffmpeg_on_path()
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(clip_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


class Transcriber:
    def __init__(self, model_size: str = WHISPER_MODEL, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = "float16" if device == "cuda" else "int8"
        log.info("loading faster-whisper %s on %s (%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: Path) -> tuple[str, str | None]:
        segments, info = self.model.transcribe(
            str(wav_path), beam_size=1, vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info.language


class VoiceDetector:
    """Wraps an HF audio-classification model that outputs fake/real labels."""

    def __init__(self, model_id: str = AI_VOICE_MODEL, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading audio-cls %s on %s", model_id, device)
        self.pipe = pipeline(
            "audio-classification",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )

    def predict(self, wav_path: Path) -> float:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return float("nan")

        chunk_n = VOICE_CHUNK_SECONDS * sr
        chunks: list[np.ndarray] = []
        for i in range(0, len(audio), chunk_n):
            ch = audio[i : i + chunk_n]
            if len(ch) >= sr:  # need at least 1s of audio
                chunks.append(ch)
        if not chunks:
            chunks = [audio]

        scores: list[float] = []
        for ch in chunks:
            out = self.pipe({"sampling_rate": sr, "raw": ch})
            ai = sum(item["score"] for item in out if not _looks_real(item["label"]))
            scores.append(float(np.clip(ai, 0.0, 1.0)))
        return float(np.mean(scores))


def _looks_real(label: str) -> bool:
    label_lc = label.lower()
    return any(hint in label_lc for hint in _REAL_HINTS)


class AudioEncoder:
    """One-stop transcript + AI-voice extractor.

    Composed of two heavy models; load once, reuse across clips. Whisper holds
    ~1-2GB VRAM at float16; the wav2vec2 classifier ~500MB. Run sequentially
    after releasing the visual encoder to stay under 10GB.
    """

    def __init__(self, device: str | None = None) -> None:
        self.transcriber = Transcriber(device=device)
        self.voice = VoiceDetector(device=device)

    def features(self, clip_path: Path) -> AudioFeatures:
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            extract_wav(clip_path, wav)
            transcript, lang = self.transcriber.transcribe(wav)
            ai_prob = self.voice.predict(wav)
        return AudioFeatures(transcript=transcript, language=lang, ai_voice_prob=ai_prob)
