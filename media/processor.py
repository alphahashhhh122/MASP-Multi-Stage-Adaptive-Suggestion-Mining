"""
media/processor.py — Real multimodal input processing for MASP pipeline.

Handles:
  1. IMAGE: Load from file path → raw bytes for Gemma3/Ollama vision captioning
  2. AUDIO: Load from file path → Whisper transcription + prosody features
  3. DATASET: Load a full dataset row → pipeline-ready input dict

Dependencies:
  pip install openai-whisper librosa soundfile numpy torch

Architecture note:
  This module sits BEFORE the LangGraph pipeline. It converts raw media files
  into the format that preprocess_node expects:
    - Images → list[bytes]     (fed to raw_images in PipelineState)
    - Audio  → transcript str  (fed to audio_transcript)
             + features dict   (fed to NEW acoustic_features field)

  The pipeline's preprocess_node then:
    - Sends images to the configured vision backend for structured captioning
    - Passes transcript + acoustic features to audio_view_builder
"""

import io
import logging
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AcousticFeatures:
    """Prosody and acoustic features extracted from audio."""

    transcript: str = ""
    duration_seconds: float = 0.0
    speaking_rate_wpm: float = 0.0  # words per minute
    avg_pitch_hz: float = 0.0
    pitch_range_hz: float = 0.0  # max - min pitch
    pitch_variability: float = 0.0  # std dev of pitch
    avg_energy_db: float = 0.0
    energy_variability: float = 0.0  # std dev of energy (RMS)
    pause_count: int = 0  # silences > 0.3s
    longest_pause_seconds: float = 0.0
    total_pause_seconds: float = 0.0
    emphasis_segments: list = field(
        default_factory=list
    )  # [{word, start, end, energy_ratio}]
    tone_classification: str = (
        "neutral"  # angry|frustrated|neutral|enthusiastic|sad|sarcastic
    )
    tone_confidence: float = 0.0
    spectral_centroid_mean: float = 0.0  # brightness of voice
    spectral_centroid_std: float = 0.0
    zero_crossing_rate: float = 0.0  # roughness indicator

    def to_description(self) -> str:
        """Convert features to natural language for the audio_view_builder prompt."""
        parts = []

        # Tone
        parts.append(f"{self.tone_classification} tone")

        # Pace
        if self.speaking_rate_wpm > 180:
            parts.append("rapid pace")
        elif self.speaking_rate_wpm < 100:
            parts.append("slow deliberate pace")
        else:
            parts.append("normal pace")

        # Emphasis
        if self.emphasis_segments:
            top = sorted(
                self.emphasis_segments,
                key=lambda x: x.get("energy_ratio", 0),
                reverse=True,
            )[:3]
            words = [seg["word"] for seg in top if "word" in seg]
            if words:
                parts.append(f"strong emphasis on {', '.join(words)}")

        # Pauses
        if self.pause_count > 3:
            parts.append(
                f"{self.pause_count} pauses (longest {self.longest_pause_seconds:.1f}s)"
            )
        elif self.longest_pause_seconds > 1.0:
            parts.append(f"notable pause of {self.longest_pause_seconds:.1f}s")

        # Pitch
        if self.pitch_variability > 50:
            parts.append("highly variable pitch (emotional)")
        elif self.pitch_variability < 15:
            parts.append("flat monotone pitch")

        # Energy
        if self.energy_variability > 0.05:
            parts.append("volume fluctuates significantly")

        return "Audio: " + ", ".join(parts) + "."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessedMedia:
    """Complete processed media ready for pipeline input."""

    raw_text: str = ""
    image_bytes_list: list = field(default_factory=list)  # list[bytes]
    audio_transcript: Optional[str] = None
    acoustic_features: Optional[AcousticFeatures] = None
    acoustic_description: Optional[str] = None  # NL description for prompt
    image_paths: list = field(default_factory=list)
    audio_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════


class ImageProcessor:
    """Load images from file paths into bytes for the vision backend."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB upload limit

    @staticmethod
    def load_image(path: str) -> Optional[bytes]:
        """Load a single image file as bytes."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Image not found: {path}")
            return None
        if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported image format: {path.suffix}")
            return None
        raw = path.read_bytes()
        if len(raw) > ImageProcessor.MAX_IMAGE_SIZE:
            logger.warning(f"Image too large ({len(raw)} bytes), resizing...")
            raw = ImageProcessor._resize_image(raw)
        return raw

    @staticmethod
    def load_images(paths: list[str]) -> list[bytes]:
        """Load multiple images."""
        results = []
        for p in paths:
            img = ImageProcessor.load_image(p)
            if img:
                results.append(img)
        return results

    @staticmethod
    def _resize_image(raw: bytes, max_dim: int = 1568) -> bytes:
        """Resize image to fit vision-backend limits."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except ImportError:
            logger.warning("Pillow not installed, returning original image")
            return raw

    @staticmethod
    def get_mime_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════


class AudioProcessor:
    """
    Transcribe audio via Whisper and extract prosody features via librosa.

    The combination of transcript + acoustic features enables the pipeline to:
    - Detect sarcasm (positive words + flat/mocking tone)
    - Identify emphasis (volume spikes on specific words)
    - Gauge frustration (pitch variability + pause patterns + speaking rate)
    - Prioritize complaints (louder segments = higher urgency)
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}

    def __init__(self, whisper_model: str = "base", device: str = "cpu"):
        """
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: 'cpu' or 'cuda'
        """
        self._whisper_model_name = whisper_model
        self._device = device
        self._whisper = None  # lazy load

    def _load_whisper(self):
        if self._whisper is None:
            import whisper

            logger.info(f"Loading Whisper model: {self._whisper_model_name}")
            self._whisper = whisper.load_model(
                self._whisper_model_name, device=self._device
            )
        return self._whisper

    def process(self, audio_path: str) -> AcousticFeatures:
        """Full audio processing: transcription + prosody extraction."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        # Step 1: Transcribe with Whisper (word-level timestamps)
        transcript, word_segments = self._transcribe(audio_path)

        # Step 2: Extract acoustic features with librosa
        features = self._extract_prosody(audio_path, word_segments)
        features.transcript = transcript

        # Step 3: Classify tone from acoustic features
        features.tone_classification, features.tone_confidence = self._classify_tone(
            features
        )

        return features

    def _transcribe(self, audio_path: str) -> tuple[str, list[dict]]:
        """Transcribe audio and get word-level timestamps."""
        model = self._load_whisper()
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language="en",
        )

        transcript = result["text"].strip()

        # Extract word segments with timestamps
        word_segments = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_segments.append(
                    {
                        "word": word_info["word"].strip(),
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "probability": word_info.get("probability", 0.0),
                    }
                )

        return transcript, word_segments

    def _extract_prosody(
        self, audio_path: str, word_segments: list[dict]
    ) -> AcousticFeatures:
        """Extract pitch, energy, pauses, emphasis from audio."""
        import librosa

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        features = AcousticFeatures(duration_seconds=duration)

        # Speaking rate
        word_count = len(word_segments)
        if duration > 0:
            features.speaking_rate_wpm = (word_count / duration) * 60

        # --- Pitch (F0) via pyin ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
            )

        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) > 0:
            features.avg_pitch_hz = float(np.mean(f0_valid))
            features.pitch_range_hz = float(np.max(f0_valid) - np.min(f0_valid))
            features.pitch_variability = float(np.std(f0_valid))

        # --- Energy (RMS) ---
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms) > 0:
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            features.avg_energy_db = float(np.mean(rms_db))
            features.energy_variability = float(np.std(rms_db))

        # --- Pauses (silence detection) ---
        frame_duration = 512 / sr  # hop_length / sr
        silence_threshold = np.percentile(rms, 10)  # bottom 10% = silence
        is_silent = rms < silence_threshold

        pauses = []
        in_pause = False
        pause_start = 0
        for i, silent in enumerate(is_silent):
            t = i * frame_duration
            if silent and not in_pause:
                in_pause = True
                pause_start = t
            elif not silent and in_pause:
                pause_dur = t - pause_start
                if pause_dur > 0.3:  # only count pauses > 300ms
                    pauses.append({"start": pause_start, "duration": pause_dur})
                in_pause = False

        features.pause_count = len(pauses)
        if pauses:
            features.longest_pause_seconds = max(p["duration"] for p in pauses)
            features.total_pause_seconds = sum(p["duration"] for p in pauses)

        # --- Emphasis detection (per-word energy analysis) ---
        if word_segments and len(rms) > 0:
            avg_rms = float(np.mean(rms))
            emphasis = []
            for ws in word_segments:
                start_frame = int(ws["start"] * sr / 512)
                end_frame = int(ws["end"] * sr / 512)
                start_frame = max(0, min(start_frame, len(rms) - 1))
                end_frame = max(start_frame + 1, min(end_frame, len(rms)))
                word_rms = float(np.mean(rms[start_frame:end_frame]))
                energy_ratio = word_rms / avg_rms if avg_rms > 0 else 1.0
                if energy_ratio > 1.4:  # 40% louder than average = emphasis
                    emphasis.append(
                        {
                            "word": ws["word"],
                            "start": ws["start"],
                            "end": ws["end"],
                            "energy_ratio": round(energy_ratio, 2),
                        }
                    )
            features.emphasis_segments = emphasis

        # --- Spectral features ---
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        if len(spectral_centroids) > 0:
            features.spectral_centroid_mean = float(np.mean(spectral_centroids))
            features.spectral_centroid_std = float(np.std(spectral_centroids))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        if len(zcr) > 0:
            features.zero_crossing_rate = float(np.mean(zcr))

        return features

    def _classify_tone(self, features: AcousticFeatures) -> tuple[str, float]:
        """
        Rule-based tone classification from acoustic features.

        This is a heuristic classifier. For the paper, you can:
        a) Keep this as-is and present it as a feature-engineered baseline
        b) Replace with a fine-tuned audio emotion model (e.g., wav2vec2-emotion)
        c) Use a local LLM/audio classifier to classify tone from the acoustic feature vector

        The key insight for the paper: the PIPELINE doesn't need perfect tone
        classification — it needs ENOUGH signal to trigger the View-Weighting
        Switch when tone contradicts text.
        """
        pitch_var = features.pitch_variability
        energy_var = features.energy_variability
        pace = features.speaking_rate_wpm
        pause_ratio = features.total_pause_seconds / max(features.duration_seconds, 0.1)
        emphasis_count = len(features.emphasis_segments)

        # Angry: high pitch variability + high energy + fast pace + emphasis
        if pitch_var > 40 and energy_var > 0.04 and pace > 160 and emphasis_count > 2:
            return "angry", min(0.9, 0.5 + pitch_var / 100)

        # Frustrated: moderate pitch var + pauses + some emphasis
        if pitch_var > 25 and pause_ratio > 0.15 and emphasis_count > 0:
            return "frustrated", min(0.85, 0.5 + pause_ratio)

        # Sarcastic: low pitch variability (flat) + slow pace + pauses
        if pitch_var < 20 and pace < 120 and pause_ratio > 0.2:
            return "sarcastic", min(0.75, 0.4 + (0.2 / max(pitch_var, 1)) * 10)

        # Sad/resigned: low energy + slow pace + many pauses
        if energy_var < 0.02 and pace < 110 and features.pause_count > 3:
            return "sad", min(0.8, 0.5 + features.pause_count * 0.05)

        # Enthusiastic: high pitch + fast pace + high energy
        if pitch_var > 35 and pace > 170 and energy_var > 0.03:
            return "enthusiastic", min(0.85, 0.5 + pace / 300)

        # Default: neutral
        return "neutral", 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MEDIA LOADER
# ═══════════════════════════════════════════════════════════════════════════════


class MediaLoader:
    """
    Load media files referenced in the MASP dataset and prepare pipeline inputs.

    Usage:
        loader = MediaLoader(audio_processor=AudioProcessor(whisper_model="base"))

        for _, row in dataset.iterrows():
            media = loader.load_from_row(row)
            result = run_pipeline(
                text=media.raw_text,
                images=media.image_bytes_list,
                audio_transcript=media.audio_transcript,
                acoustic_features=media.acoustic_features,
            )
    """

    def __init__(
        self,
        images_dir: str = "masp_images",
        audio_dir: str = "masp_audio",
        audio_processor: Optional[AudioProcessor] = None,
    ):
        self.images_dir = Path(images_dir)
        self.audio_dir = Path(audio_dir)
        self.audio_processor = audio_processor or AudioProcessor()
        self.img_processor = ImageProcessor()

    def load_from_row(self, row: dict) -> ProcessedMedia:
        """Load all media for a single dataset row."""
        media = ProcessedMedia(
            raw_text=str(row.get("raw_text", "")),
            metadata={
                "entry_id": row.get("entry_id"),
                "domain": row.get("domain"),
                "extraction_path": row.get("extraction_path"),
            },
        )

        # Load image if path exists
        image_path = row.get("image_path", "—")
        if image_path and image_path != "—":
            full_path = self._resolve_path(image_path, self.images_dir)
            if full_path:
                img_bytes = ImageProcessor.load_image(str(full_path))
                if img_bytes:
                    media.image_bytes_list = [img_bytes]
                    media.image_paths = [str(full_path)]

        # Also try convention: {entry_id}.{ext} in images_dir (with underscore normalization)
        if not media.image_bytes_list:
            entry_id = row.get("entry_id", "").replace("-", "_")
            for ext in ImageProcessor.SUPPORTED_FORMATS:
                candidate = self.images_dir / f"{entry_id}{ext}"
                if candidate.exists():
                    img_bytes = ImageProcessor.load_image(str(candidate))
                    if img_bytes:
                        media.image_bytes_list = [img_bytes]
                        media.image_paths = [str(candidate)]
                    break

        # Load and process audio if path exists
        audio_path = row.get("audio_path", "—")
        if audio_path and audio_path != "—":
            full_path = self._resolve_path(audio_path, self.audio_dir)
            if full_path:
                try:
                    features = self.audio_processor.process(str(full_path))
                    media.audio_transcript = features.transcript
                    media.acoustic_features = features
                    media.acoustic_description = features.to_description()
                    media.audio_path = str(full_path)
                except Exception as e:
                    logger.error(f"Audio processing failed for {audio_path}: {e}")

        # Also try convention: {entry_id}.{ext} in audio_dir (with underscore normalization)
        if not media.audio_transcript:
            entry_id = row.get("entry_id", "").replace("-", "_")
            for ext in AudioProcessor.SUPPORTED_FORMATS:
                candidate = self.audio_dir / f"{entry_id}{ext}"
                if candidate.exists():
                    try:
                        features = self.audio_processor.process(str(candidate))
                        media.audio_transcript = features.transcript
                        media.acoustic_features = features
                        media.acoustic_description = features.to_description()
                        media.audio_path = str(candidate)
                    except Exception as e:
                        logger.error(f"Audio processing failed for {candidate}: {e}")
                    break

        # Fallback: if dataset has multimodal_context with audio description but no file
        if not media.audio_transcript and row.get("modality") in (
            "audio_transcript",
            "multimodal",
        ):
            mm_ctx = str(row.get("multimodal_context", ""))
            if "Audio:" in mm_ctx:
                # Extract the audio description as a synthetic acoustic feature set
                media.acoustic_description = mm_ctx[mm_ctx.index("Audio:") :]
                # Use raw_text as transcript (the audio review text IS the transcript)
                media.audio_transcript = media.raw_text

        return media

    def _resolve_path(self, path_str: str, default_dir: Path) -> Optional[Path]:
        """Resolve a path, trying absolute then relative to default_dir."""
        p = Path(path_str)
        if p.exists():
            return p
        relative = default_dir / p.name
        if relative.exists():
            return relative
        return None

    def validate_media(self, dataset_path: str) -> dict:
        """Check which media files exist for the dataset."""
        import pandas as pd

        df = pd.read_excel(dataset_path, sheet_name="MASP Dataset v3")

        stats = {
            "total": len(df),
            "needs_image": 0,
            "has_image": 0,
            "missing_image": 0,
            "needs_audio": 0,
            "has_audio": 0,
            "missing_audio": 0,
            "missing_images": [],
            "missing_audio_files": [],
        }

        for _, row in df.iterrows():
            mod = row.get("modality", "")
            eid = row.get("entry_id", "")

            # Check images
            if mod in ("image_review", "multimodal"):
                stats["needs_image"] += 1
                found = False
                eid_normalized = eid.replace("-", "_")
                for ext in ImageProcessor.SUPPORTED_FORMATS:
                    if (self.images_dir / f"{eid_normalized}{ext}").exists():
                        found = True
                        break
                if found:
                    stats["has_image"] += 1
                else:
                    stats["missing_image"] += 1
                    stats["missing_images"].append(eid)

            # Check audio
            if mod in ("audio_transcript", "multimodal"):
                stats["needs_audio"] += 1
                found = False
                eid_normalized = eid.replace("-", "_")
                for ext in AudioProcessor.SUPPORTED_FORMATS:
                    if (self.audio_dir / f"{eid_normalized}{ext}").exists():
                        found = True
                        break
                if found:
                    stats["has_audio"] += 1
                else:
                    stats["missing_audio"] += 1
                    stats["missing_audio_files"].append(eid)

        return stats
