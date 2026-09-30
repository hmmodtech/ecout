"""
Speech-to-text backends.

* ``FasterWhisperTranscriber`` – local, CPU/GPU, tuned for latency
  (greedy decoding, int8 on CPU, Silero VAD, numpy input → no temp files,
  no ffmpeg).
* ``OpenAIWhisperTranscriber`` – OpenAI ``whisper-1`` / ``gpt-4o-*-transcribe``.
* ``GeminiAudioTranscriber`` – Gemini multimodal transcription (uses the
  same API key as the assistant).

All backends expose ``transcribe(audio: np.ndarray[float32 @16 kHz]) -> str``
plus the legacy ``get_transcription(wav_path)`` used by older code.
"""
from __future__ import annotations

import io
import os
import re
import time
import wave
from typing import Optional

import numpy as np

_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thanks", "thanks.", "bye", "bye.", "thank you very much",
    "thanks for watching", "thanks for watching.", "thank you for watching", "thank you for watching.",
    "subtitles by the amara.org community", "the end", "the end.", "so", "so.", "oh", "oh.", "hmm", "hmm.",
    "uh", "um", ".", "..", "...", "you.", "okay.", "okay", "yeah.", "yeah", "mm-hmm.", "mm-hmm",
    "i'm sorry.", "peace.", "music", "[music]", "(music)", "[blank_audio]", "[silence]",
}


def clean_transcript(text: str) -> str:
    """Normalise whitespace and drop the classic Whisper silence hallucinations."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    low = t.lower()
    if low in _HALLUCINATIONS:
        return ""
    if re.fullmatch(r"[\W_]+", t):
        return ""
    # repeated single token ("you you you you")
    words = low.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return ""
    return t


def float_to_wav_bytes(audio: np.ndarray, rate: int = 16000) -> bytes:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def load_wav_as_float(path: str) -> np.ndarray:
    """Read a PCM WAV file → mono float32 @16 kHz (for the legacy API)."""
    from audio_engine import pcm16_to_mono_float, resample_to_16k

    with wave.open(path, "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        data = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported")
    return resample_to_16k(pcm16_to_mono_float(data, channels), rate)


class BaseTranscriber:
    name = "base"
    language: Optional[str] = "en"

    def transcribe(self, audio: np.ndarray) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def get_transcription(self, wav_file_path: str) -> str:
        try:
            return self.transcribe(load_wav_as_float(wav_file_path))
        except Exception as exc:
            print(f"[{self.name}] {exc}")
            return ""

    def set_vocabulary(self, words: str):
        """Optional domain vocabulary hint (acronyms, names)."""


# --------------------------------------------------------------------------- #
# faster-whisper
# --------------------------------------------------------------------------- #
class FasterWhisperTranscriber(BaseTranscriber):
    name = "faster-whisper"

    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str = "auto",
                 language: Optional[str] = "en", beam_size: int = 1, cpu_threads: int = 0,
                 num_workers: int = 2, vad_filter: bool = True, vocabulary: str = ""):
        from faster_whisper import WhisperModel  # heavy import, keep local

        if device == "auto":
            device = "cpu"
            try:
                import ctranslate2  # type: ignore

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        if cpu_threads <= 0:
            cpu_threads = max(2, (os.cpu_count() or 4) // 2)
        if language and not language.startswith("en") and model_size.endswith(".en"):
            model_size = model_size[:-3]  # multilingual variant
        t0 = time.perf_counter()
        print(f"[faster-whisper] loading '{model_size}' on {device} ({compute_type}) …")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, num_workers=max(1, num_workers))
        print(f"[faster-whisper] ready in {time.perf_counter() - t0:.1f}s")
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self.language = language if language and language != "auto" else None
        self.beam_size = max(1, int(beam_size))
        self.vad_filter = vad_filter
        self.vocabulary = vocabulary or ""

    def set_vocabulary(self, words: str):
        self.vocabulary = (words or "")[:200]

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < 1600:  # < 0.1 s
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        kwargs = dict(
            language=self.language,
            beam_size=self.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=self.vad_filter,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200, "threshold": 0.4},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        if self.vocabulary:
            kwargs["hotwords"] = self.vocabulary
        try:
            segments, _info = self.model.transcribe(audio, **kwargs)
            texts = []
            for seg in segments:
                t = (seg.text or "").strip()
                if not t:
                    continue
                if getattr(seg, "no_speech_prob", 0.0) > 0.85 and getattr(seg, "avg_logprob", 0.0) < -1.0:
                    continue
                texts.append(t)
            return clean_transcript(" ".join(texts))
        except Exception as exc:
            print(f"[faster-whisper] transcription error: {exc}")
            return ""


# --------------------------------------------------------------------------- #
# OpenAI API
# --------------------------------------------------------------------------- #
class OpenAIWhisperTranscriber(BaseTranscriber):
    name = "openai-api"

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1", language: Optional[str] = "en"):
        from openai import OpenAI

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                try:
                    import keys  # type: ignore

                    api_key = getattr(keys, "OPENAI_API_KEY", None)
                except Exception:
                    api_key = None
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language if language and language != "auto" else None

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < 1600:
            return ""
        try:
            wav = float_to_wav_bytes(audio)
            kwargs = {"model": self.model, "file": ("speech.wav", wav, "audio/wav")}
            if self.language:
                kwargs["language"] = self.language
            result = self.client.audio.transcriptions.create(**kwargs)
            return clean_transcript(getattr(result, "text", "") or "")
        except Exception as exc:
            print(f"[openai-api] {exc}")
            return ""


# --------------------------------------------------------------------------- #
# Gemini audio
# --------------------------------------------------------------------------- #
class GeminiAudioTranscriber(BaseTranscriber):
    name = "gemini-audio"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", language: Optional[str] = "en"):
        from llm_client import _load_sdks

        new_sdk, old_sdk = _load_sdks()
        self.language = language
        self.model = model
        if new_sdk:
            genai, types = new_sdk
            self._types = types
            self.client = genai.Client(api_key=api_key)
            self._backend = "new"
        elif old_sdk:
            old_sdk.configure(api_key=api_key)
            self.client = old_sdk.GenerativeModel(model)
            self._backend = "old"
        else:
            raise RuntimeError("No Gemini SDK installed")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < 1600:
            return ""
        wav = float_to_wav_bytes(audio)
        prompt = "Transcribe this audio verbatim. Output only the spoken words, nothing else. If there is no speech output nothing."
        try:
            if self._backend == "new":
                part = self._types.Part.from_bytes(data=wav, mime_type="audio/wav")
                resp = self.client.models.generate_content(model=self.model, contents=[prompt, part])
            else:
                resp = self.client.generate_content([prompt, {"mime_type": "audio/wav", "data": wav}])
            return clean_transcript(getattr(resp, "text", "") or "")
        except Exception as exc:
            print(f"[gemini-audio] {exc}")
            return ""


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_model(use_api: bool = False, config: Optional[dict] = None) -> BaseTranscriber:
    """Build the transcription backend selected by the config / CLI flag."""
    cfg = config or {}
    lang_mode = str(cfg.get("language_mode", "English Only"))
    language = "en" if lang_mode.lower().startswith("english") else None
    if use_api or str(cfg.get("asr_backend", "local")).lower() == "api":
        provider = str(cfg.get("asr_api_provider", "openai")).lower()
        if provider == "gemini":
            from llm_client import resolve_api_key

            return GeminiAudioTranscriber(resolve_api_key(cfg), cfg.get("asr_api_model", cfg.get("model_name", "gemini-2.5-flash")), language)
        return OpenAIWhisperTranscriber(model=cfg.get("asr_api_model", "whisper-1"), language=language)
    return FasterWhisperTranscriber(
        model_size=str(cfg.get("whisper_model", "base.en")),
        device=str(cfg.get("whisper_device", "auto")),
        compute_type=str(cfg.get("whisper_compute_type", "auto")),
        language=language,
        beam_size=int(cfg.get("asr_beam_size", 1)),
        cpu_threads=int(cfg.get("asr_cpu_threads", 0)),
        num_workers=int(cfg.get("asr_workers", 2)),
        vad_filter=bool(cfg.get("asr_vad_filter", True)),
    )
