"""
Audio capture engine with automatic device switching.

Replaces the old ``AudioRecorder.py`` / ``custom_speech_recognition`` pair.

Key properties
--------------
* Captures the microphone and the system output (WASAPI loopback) with
  ``pyaudiowpatch`` in callback mode – no blocking reads that hang when a
  device disappears.
* A **watchdog** thread polls the Windows default endpoints (through the
  Core Audio API – pycaw if installed, otherwise a tiny ctypes shim) and the
  health of the open streams.  When the default mic/speaker changes, a
  Bluetooth headset connects, or a stream dies, both streams are re-opened
  on the now-active devices.  The transcription pipeline keeps running: the
  queue of speech *segments* is format-agnostic (always 16 kHz mono float32).
* Energy based phrase segmentation with a short pause threshold and
  optional *interim* segments, so a question reaches the recogniser a few
  hundred milliseconds after the interviewer stops talking.

The pure logic (segmenter, resampler, device selection) is separated from
the PortAudio glue so it can be unit tested on any OS with a fake backend.
"""
from __future__ import annotations

import collections
import itertools
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TARGET_RATE = 16000
SOURCES = ("You", "Speaker")  # "You" = microphone, "Speaker" = system output / meeting audio


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
def pcm16_to_mono_float(pcm: bytes, channels: int) -> np.ndarray:
    """Interleaved int16 PCM → mono float32 in [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        n = (len(x) // channels) * channels
        x = x[:n].reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        x = x.astype(np.float32)
    return x / 32768.0


def rms_int16(x: np.ndarray) -> float:
    """RMS energy expressed on the int16 scale (comparable to SpeechRecognition thresholds)."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float32))) * 32768.0)


def resample_to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    """Cheap but adequate resampling of mono float32 audio to 16 kHz."""
    if x.size == 0:
        return x.astype(np.float32)
    if rate == TARGET_RATE:
        return x.astype(np.float32, copy=False)
    if rate > TARGET_RATE and rate % TARGET_RATE == 0:
        factor = rate // TARGET_RATE
        n = (len(x) // factor) * factor
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return x[:n].reshape(-1, factor).mean(axis=1).astype(np.float32)
    # generic path: light anti-alias (moving average) + linear interpolation
    y = x.astype(np.float32)
    if rate > TARGET_RATE:
        w = int(math.ceil(rate / TARGET_RATE))
        if w > 1:
            kernel = np.ones(w, dtype=np.float32) / w
            y = np.convolve(y, kernel, mode="same")
    n_out = max(1, int(round(len(x) * TARGET_RATE / rate)))
    t_out = np.arange(n_out, dtype=np.float64) * (rate / TARGET_RATE)
    return np.interp(t_out, np.arange(len(y), dtype=np.float64), y).astype(np.float32)


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
_phrase_counter = itertools.count(1)


@dataclass
class Segment:
    source: str
    audio: np.ndarray            # float32 mono @16 kHz
    start_time: float            # wall clock (time.time())
    end_time: float
    is_final: bool
    phrase_id: int
    rate: int = TARGET_RATE

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.rate)


class PhraseSegmenter:
    """Energy based voice-activity segmentation (port of the SpeechRecognition
    ``listen`` algorithm, extended with hard cuts and interim segments)."""

    def __init__(self, sample_rate: int, energy_threshold: float = 300.0,
                 dynamic_energy: bool = True, dynamic_damping: float = 0.15,
                 dynamic_ratio: float = 1.5, min_energy_threshold: float = 120.0,
                 max_energy_threshold: float = 6000.0,
                 pause_threshold: float = 0.5, phrase_threshold: float = 0.3,
                 non_speaking_duration: float = 0.2, phrase_time_limit: float = 8.0,
                 interim_interval: float = 2.5, source: str = "Speaker"):
        self.sample_rate = int(sample_rate)
        self.threshold = float(energy_threshold)
        self.dynamic_energy = dynamic_energy
        self.dynamic_damping = dynamic_damping
        self.dynamic_ratio = dynamic_ratio
        self.min_energy_threshold = float(min_energy_threshold)
        self.max_energy_threshold = float(max_energy_threshold)
        self.pause_threshold = float(pause_threshold)
        self.phrase_threshold = float(phrase_threshold)
        self.non_speaking_duration = float(min(non_speaking_duration, pause_threshold))
        self.phrase_time_limit = float(phrase_time_limit)
        self.interim_interval = float(interim_interval) if interim_interval else 0.0
        self.source = source
        self._reset_phrase()
        self.prebuffer: collections.deque = collections.deque()
        self.prebuffer_dur = 0.0
        self.last_energy = 0.0

    def _reset_phrase(self):
        self.speaking = False
        self.frames: List[np.ndarray] = []
        self.frames_dur = 0.0
        self.speech_dur = 0.0
        self.silence_run = 0.0
        self.since_interim = 0.0
        self.phrase_id = 0
        self.phrase_wall_start = 0.0

    # -- internals -------------------------------------------------------------
    def _emit(self, is_final: bool, now: float, trim_trailing: float = 0.0) -> Optional[Segment]:
        frames = self.frames
        if trim_trailing > 0 and frames:
            keep = self.frames_dur - trim_trailing
            acc = 0.0
            kept: List[np.ndarray] = []
            for fr in frames:
                if acc >= keep:
                    break
                kept.append(fr)
                acc += len(fr) / self.sample_rate
            frames = kept
        if not frames:
            return None
        native = np.concatenate(frames)
        audio = resample_to_16k(native, self.sample_rate)
        duration = len(native) / self.sample_rate
        end_time = now - max(0.0, self.frames_dur - duration)
        return Segment(self.source, audio, end_time - duration, end_time, is_final, self.phrase_id)

    def feed(self, chunk: np.ndarray, now: Optional[float] = None) -> List[Segment]:
        """Feed a mono float32 chunk at the native rate.  Returns 0..2 segments."""
        now = time.time() if now is None else now
        out: List[Segment] = []
        if chunk.size == 0:
            return out
        dt = len(chunk) / self.sample_rate
        energy = rms_int16(chunk)
        self.last_energy = energy

        if not self.speaking:
            self.prebuffer.append(chunk)
            self.prebuffer_dur += dt
            while self.prebuffer_dur - (len(self.prebuffer[0]) / self.sample_rate) >= self.non_speaking_duration and len(self.prebuffer) > 1:
                self.prebuffer_dur -= len(self.prebuffer.popleft()) / self.sample_rate
            if energy > self.threshold:
                self.speaking = True
                self.phrase_id = next(_phrase_counter)
                self.frames = list(self.prebuffer)
                self.frames_dur = self.prebuffer_dur
                self.phrase_wall_start = now - self.prebuffer_dur
                self.speech_dur = dt
                self.silence_run = 0.0
                self.since_interim = 0.0
                self.prebuffer.clear()
                self.prebuffer_dur = 0.0
            elif self.dynamic_energy:
                damping = self.dynamic_damping ** dt
                target = energy * self.dynamic_ratio
                self.threshold = self.threshold * damping + target * (1 - damping)
                self.threshold = min(max(self.threshold, self.min_energy_threshold), self.max_energy_threshold)
            return out

        # currently inside a phrase
        self.frames.append(chunk)
        self.frames_dur += dt
        self.since_interim += dt
        if energy > self.threshold:
            self.silence_run = 0.0
            self.speech_dur += dt
        else:
            self.silence_run += dt

        if self.silence_run >= self.pause_threshold:
            if self.speech_dur >= self.phrase_threshold:
                seg = self._emit(True, now, trim_trailing=max(0.0, self.silence_run - self.non_speaking_duration))
                if seg is not None:
                    out.append(seg)
            self._reset_phrase()
            return out

        if self.frames_dur >= self.phrase_time_limit:
            seg = self._emit(True, now)
            if seg is not None:
                out.append(seg)
            # continue immediately with a fresh phrase (continuous speech)
            pid = next(_phrase_counter)
            self._reset_phrase()
            self.speaking = True
            self.phrase_id = pid
            self.phrase_wall_start = now
            self.speech_dur = 0.0
            return out

        if self.interim_interval and self.since_interim >= self.interim_interval and self.speech_dur >= 0.8:
            seg = self._emit(False, now)
            if seg is not None:
                out.append(seg)
            self.since_interim = 0.0
        return out

    def flush(self, now: Optional[float] = None) -> List[Segment]:
        """Finalise whatever is buffered (device switch / shutdown)."""
        now = time.time() if now is None else now
        out: List[Segment] = []
        if self.speaking and self.speech_dur >= self.phrase_threshold:
            seg = self._emit(True, now)
            if seg is not None:
                out.append(seg)
        self._reset_phrase()
        self.prebuffer.clear()
        self.prebuffer_dur = 0.0
        return out


# --------------------------------------------------------------------------- #
# Windows default-endpoint probe (Core Audio) – detects device changes that
# PortAudio cannot see while streams are open.
# --------------------------------------------------------------------------- #
E_RENDER, E_CAPTURE = 0, 1
ROLE_CONSOLE, ROLE_MULTIMEDIA, ROLE_COMMUNICATIONS = 0, 1, 2


class DefaultDeviceProbe:
    """Returns the IDs / names of the current default audio endpoints.

    ``snapshot()`` → ``{"render_mm": (id, name), "render_comm": (...),
    "capture_mm": (...), "capture_comm": (...)}`` or ``None`` when no backend
    is usable (non-Windows, missing libraries).  Must be used from a single
    thread (COM apartment rules); :class:`AudioEngine` calls it from the
    watchdog thread only.
    """

    def __init__(self):
        self.backend: Optional[str] = None
        self._initialised = False
        self._pycaw: Dict[str, Any] = {}
        self._ole32 = None

    # -- initialisation (lazy, in the calling thread) --------------------------
    def _init(self):
        self._initialised = True
        if not sys.platform.startswith("win"):
            return
        try:
            import comtypes  # type: ignore

            try:
                comtypes.CoInitialize()
            except Exception:
                pass
            try:
                from pycaw.api.mmdeviceapi import IMMDeviceEnumerator  # type: ignore
                from pycaw.constants import CLSID_MMDeviceEnumerator  # type: ignore
                from pycaw.utils import AudioUtilities  # type: ignore
            except Exception:
                from pycaw.pycaw import AudioUtilities, CLSID_MMDeviceEnumerator, IMMDeviceEnumerator  # type: ignore
            self._pycaw = {
                "comtypes": comtypes,
                "IMMDeviceEnumerator": IMMDeviceEnumerator,
                "CLSID": CLSID_MMDeviceEnumerator,
                "AudioUtilities": AudioUtilities,
            }
            # smoke test
            self._pycaw_endpoint(E_RENDER, ROLE_MULTIMEDIA)
            self.backend = "pycaw"
            return
        except Exception:
            self._pycaw = {}
        try:
            self._init_ctypes()
            self.backend = "ctypes"
        except Exception:
            self.backend = None

    # -- pycaw backend ----------------------------------------------------------
    def _pycaw_endpoint(self, flow: int, role: int) -> Tuple[Optional[str], Optional[str]]:
        comtypes = self._pycaw["comtypes"]
        try:
            enumerator = comtypes.CoCreateInstance(
                self._pycaw["CLSID"], self._pycaw["IMMDeviceEnumerator"], comtypes.CLSCTX_INPROC_SERVER)
            dev = enumerator.GetDefaultAudioEndpoint(flow, role)
        except Exception:
            return None, None
        if dev is None:
            return None, None
        dev_id = None
        try:
            dev_id = str(dev.GetId())
        except Exception:
            pass
        name = None
        try:
            wrapped = self._pycaw["AudioUtilities"].CreateDevice(dev)
            name = getattr(wrapped, "FriendlyName", None)
        except Exception:
            pass
        return dev_id, name

    # -- ctypes backend (no extra dependencies) --------------------------------
    def _init_ctypes(self):
        import ctypes

        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
        ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED; RPC_E_CHANGED_MODE is fine
        self._ole32 = ole32

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                        ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

        def guid(s: str) -> GUID:
            g = GUID()
            if ole32.CLSIDFromString(ctypes.c_wchar_p(s), ctypes.byref(g)) != 0:
                raise OSError("CLSIDFromString failed")
            return g

        self._clsid = guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
        self._iid = guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        self._ctypes = ctypes
        # smoke test
        self._ctypes_endpoint(E_RENDER, ROLE_MULTIMEDIA)

    def _vtbl(self, obj, index: int, restype, *argtypes):
        ctypes = self._ctypes
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
        return proto(vtable[index])

    def _ctypes_endpoint(self, flow: int, role: int) -> Tuple[Optional[str], Optional[str]]:
        ctypes = self._ctypes
        ole32 = self._ole32
        enumerator = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(self._clsid), None, 1, ctypes.byref(self._iid), ctypes.byref(enumerator))
        if hr != 0 or not enumerator:
            return None, None
        dev_id = None
        try:
            get_default = self._vtbl(enumerator, 4, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
            dev = ctypes.c_void_p()
            hr = get_default(enumerator, flow, role, ctypes.byref(dev))
            if hr == 0 and dev:
                try:
                    get_id = self._vtbl(dev, 5, ctypes.c_long, ctypes.POINTER(ctypes.c_wchar_p))
                    pid = ctypes.c_wchar_p()
                    if get_id(dev, ctypes.byref(pid)) == 0 and pid:
                        dev_id = str(pid.value)
                        ole32.CoTaskMemFree(ctypes.cast(pid, ctypes.c_void_p))
                finally:
                    self._vtbl(dev, 2, ctypes.c_ulong)(dev)  # Release
        finally:
            self._vtbl(enumerator, 2, ctypes.c_ulong)(enumerator)  # Release
        return dev_id, None

    # -- public -----------------------------------------------------------------
    def snapshot(self) -> Optional[Dict[str, Tuple[Optional[str], Optional[str]]]]:
        if not self._initialised:
            self._init()
        if self.backend == "pycaw":
            fn = self._pycaw_endpoint
        elif self.backend == "ctypes":
            fn = self._ctypes_endpoint
        else:
            return None
        try:
            return {
                "render_mm": fn(E_RENDER, ROLE_MULTIMEDIA),
                "render_comm": fn(E_RENDER, ROLE_COMMUNICATIONS),
                "capture_mm": fn(E_CAPTURE, ROLE_MULTIMEDIA),
                "capture_comm": fn(E_CAPTURE, ROLE_COMMUNICATIONS),
            }
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# Device selection (works with the real pyaudiowpatch module or a fake)
# --------------------------------------------------------------------------- #
@dataclass
class DeviceSpec:
    index: int
    name: str
    channels: int
    rate: int
    kind: str  # "mic" | "loopback"


def _wasapi_info(p, pyaudio_module):
    try:
        return p.get_host_api_info_by_type(pyaudio_module.paWASAPI)
    except Exception:
        return None


def _name_matches(target: Optional[str], name: Optional[str]) -> bool:
    if not target or not name:
        return False
    t, n = target.strip().lower(), name.strip().lower()
    return n.startswith(t) or t.startswith(n.replace(" [loopback]", "")) or t in n


def resolve_mic_candidates(p, pyaudio_module, preferred_name: Optional[str] = None) -> List[DeviceSpec]:
    """Ordered list of microphone devices to try (best first)."""
    cands: List[DeviceSpec] = []
    seen = set()

    def add(info, kind="mic"):
        try:
            idx = int(info["index"])
            if idx in seen or int(info.get("maxInputChannels", 0)) <= 0:
                return
            seen.add(idx)
            cands.append(DeviceSpec(idx, str(info.get("name", f"device {idx}")),
                                    int(info.get("maxInputChannels", 1)),
                                    int(info.get("defaultSampleRate", 44100) or 44100), kind))
        except Exception:
            pass

    wasapi = _wasapi_info(p, pyaudio_module)
    if preferred_name and wasapi:
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("hostApi") == wasapi.get("index") and not info.get("isLoopbackDevice") \
                        and int(info.get("maxInputChannels", 0)) > 0 and _name_matches(preferred_name, info.get("name")):
                    add(info)
                    break
        except Exception:
            pass
    if wasapi and wasapi.get("defaultInputDevice", -1) is not None and int(wasapi.get("defaultInputDevice", -1)) >= 0:
        try:
            add(p.get_device_info_by_index(int(wasapi["defaultInputDevice"])))
        except Exception:
            pass
    try:
        add(p.get_default_input_device_info())
    except Exception:
        pass
    return cands


def resolve_loopback_candidates(p, pyaudio_module, preferred_output_name: Optional[str] = None) -> List[DeviceSpec]:
    """Ordered list of WASAPI loopback devices to try (best first)."""
    cands: List[DeviceSpec] = []
    seen = set()

    def add(info):
        try:
            idx = int(info["index"])
            if idx in seen:
                return
            seen.add(idx)
            cands.append(DeviceSpec(idx, str(info.get("name", f"device {idx}")),
                                    max(1, int(info.get("maxInputChannels", 2) or 2)),
                                    int(info.get("defaultSampleRate", 48000) or 48000), "loopback"))
        except Exception:
            pass

    loopbacks: List[dict] = []
    try:
        loopbacks = list(p.get_loopback_device_info_generator())
    except Exception:
        loopbacks = []

    target_names: List[str] = []
    if preferred_output_name:
        target_names.append(preferred_output_name)
    wasapi = _wasapi_info(p, pyaudio_module)
    if wasapi is not None:
        try:
            out_idx = int(wasapi.get("defaultOutputDevice", -1))
            if out_idx >= 0:
                default_out = p.get_device_info_by_index(out_idx)
                if default_out.get("isLoopbackDevice"):
                    add(default_out)
                else:
                    target_names.append(str(default_out.get("name", "")))
        except Exception:
            pass
    for target in target_names:
        for lb in loopbacks:
            if _name_matches(target, lb.get("name")):
                add(lb)
                break
    try:
        add(p.get_default_wasapi_loopback())
    except Exception:
        pass
    # last resort: any loopback at all
    for lb in loopbacks:
        add(lb)
    return cands


def open_specs_for(spec: DeviceSpec) -> List[Tuple[int, int]]:
    """(channels, rate) combinations to try for a device, best first."""
    combos: List[Tuple[int, int]] = []
    if spec.kind == "mic":
        combos += [(1, spec.rate), (spec.channels, spec.rate), (1, 16000), (1, 44100), (1, 48000)]
    else:
        combos += [(spec.channels, spec.rate), (1, spec.rate), (2, spec.rate), (2, 48000), (2, 44100)]
    out: List[Tuple[int, int]] = []
    for c in combos:
        if c[0] > 0 and c[1] > 0 and c not in out:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class OpenStream:
    source: str
    spec: DeviceSpec
    channels: int
    rate: int
    stream: Any
    generation: int
    last_callback: float = field(default_factory=time.time)
    callbacks: int = 0


class AudioEngine:
    """Owns the PortAudio instance, the two capture streams, the per-source
    segmenters and the device watchdog."""

    def __init__(self, config: Optional[dict] = None,
                 on_segment: Optional[Callable[[Segment], None]] = None,
                 on_status: Optional[Callable[[str, str], None]] = None,
                 pyaudio_module=None, probe: Optional[DefaultDeviceProbe] = None,
                 sleep=time.sleep, clock=time.time):
        self.config = config or {}
        self.on_segment = on_segment or (lambda seg: None)
        self.on_status = on_status or (lambda level, msg: None)
        self._pyaudio = pyaudio_module
        self._probe = probe if probe is not None else DefaultDeviceProbe()
        self._sleep = sleep
        self._clock = clock

        self._p = None
        self._streams: Dict[str, OpenStream] = {}
        self._raw_queues: Dict[str, "queue.Queue"] = {s: queue.Queue() for s in SOURCES}
        self._segmenters: Dict[str, Optional[PhraseSegmenter]] = {s: None for s in SOURCES}
        self._formats: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._generation = 0
        self._lock = threading.RLock()
        self._running = False
        self._threads: List[threading.Thread] = []
        self._levels: Dict[str, float] = {s: 0.0 for s in SOURCES}
        self._last_snapshot = None
        self._pending_snapshot = None
        self._pending_since = 0.0
        self._last_open_attempt = 0.0
        self._retry_delay = 1.0
        self.device_names: Dict[str, str] = {s: "—" for s in SOURCES}
        self.reconnect_count = 0
        self.last_error: Optional[str] = None
        self.enabled_sources = {
            "You": bool(self.config.get("transcribe_mic", True)),
            "Speaker": True,
        }

    # -- configuration helpers -------------------------------------------------
    def _make_segmenter(self, source: str, rate: int) -> PhraseSegmenter:
        cfg = self.config
        is_mic = source == "You"
        return PhraseSegmenter(
            sample_rate=rate,
            energy_threshold=float(cfg.get("energy_threshold", 300)),
            dynamic_energy=bool(cfg.get("dynamic_energy", True)),
            min_energy_threshold=float(cfg.get("min_energy_threshold", 120 if is_mic else 100)),
            pause_threshold=float(cfg.get("pause_threshold", 0.5)),
            phrase_threshold=float(cfg.get("phrase_threshold", 0.3)),
            non_speaking_duration=float(cfg.get("non_speaking_duration", 0.2)),
            phrase_time_limit=float(cfg.get("phrase_time_limit", 8.0)),
            interim_interval=float(cfg.get("interim_interval", 2.5)) if not is_mic or cfg.get("mic_interim", False) else 0.0,
            source=source,
        )

    def _pa(self):
        if self._pyaudio is None:
            import pyaudiowpatch as pyaudio  # type: ignore

            self._pyaudio = pyaudio
        return self._pyaudio

    # -- lifecycle ----------------------------------------------------------------
    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        for source in SOURCES:
            t = threading.Thread(target=self._process_loop, args=(source,), name=f"audio-proc-{source}", daemon=True)
            t.start()
            self._threads.append(t)
        self._open_all("startup")
        wd = threading.Thread(target=self._watchdog_loop, name="audio-watchdog", daemon=True)
        wd.start()
        self._threads.append(wd)

    def stop(self):
        with self._lock:
            self._running = False
            self._close_all()
        for source in SOURCES:
            self._raw_queues[source].put(None)

    def reconnect(self, reason: str = "manual"):
        """Re-open both streams on the currently active default devices."""
        with self._lock:
            if not self._running:
                return
            self._close_all()
            self._sleep(0.15)  # let WASAPI settle
            self._open_all(reason)

    def levels(self) -> Dict[str, float]:
        return dict(self._levels)

    def status_text(self) -> str:
        mic = self.device_names.get("You", "—")
        spk = self.device_names.get("Speaker", "—")
        return f"🎤 {mic}   |   🔊 {spk}"

    # -- opening / closing streams ---------------------------------------------
    def _snapshot_names(self) -> Tuple[Optional[str], Optional[str]]:
        """Preferred (mic_name, output_name) according to the device_role setting."""
        snap = self._last_snapshot or {}
        role = str(self.config.get("device_role", "auto")).lower()

        def pick(mm_key, comm_key):
            mm = snap.get(mm_key, (None, None))
            comm = snap.get(comm_key, (None, None))
            if role == "communications":
                return comm[1] or mm[1]
            if role == "multimedia":
                return mm[1]
            # auto: communications device wins when it differs (Teams/Zoom use it)
            if comm[0] and mm[0] and comm[0] != mm[0]:
                return comm[1] or mm[1]
            return mm[1] or comm[1]

        return pick("capture_mm", "capture_comm"), pick("render_mm", "render_comm")

    def _open_all(self, reason: str):
        with self._lock:
            self._last_open_attempt = self._clock()
            pyaudio = self._pa()
            try:
                self._p = pyaudio.PyAudio()
            except Exception as exc:
                self.last_error = f"PortAudio init failed: {exc}"
                self.on_status("error", self.last_error)
                return
            self._generation += 1
            gen = self._generation
            if self._last_snapshot is None:
                try:
                    self._last_snapshot = self._probe.snapshot()
                except Exception:
                    self._last_snapshot = None
            mic_pref, out_pref = self._snapshot_names()
            opened = []
            failed = []
            if self.enabled_sources.get("You", True):
                if self._open_source("You", resolve_mic_candidates(self._p, pyaudio, mic_pref), gen):
                    opened.append("You")
                else:
                    failed.append("mic")
            if self.enabled_sources.get("Speaker", True):
                if self._open_source("Speaker", resolve_loopback_candidates(self._p, pyaudio, out_pref), gen):
                    opened.append("Speaker")
                else:
                    failed.append("speaker")
            if opened:
                self._retry_delay = 1.0
                self.on_status("info", f"Audio ready ({reason}): {self.status_text()}")
            if failed:
                self.last_error = f"Could not open {', '.join(failed)} – will retry"
                self.on_status("warning", self.last_error)
                self._retry_delay = min(self._retry_delay * 2, 10.0)

    def _open_source(self, source: str, candidates: Sequence[DeviceSpec], gen: int) -> bool:
        pyaudio = self._pa()
        for spec in candidates:
            for channels, rate in open_specs_for(spec):
                try:
                    stream = self._p.open(
                        format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                        input_device_index=spec.index,
                        frames_per_buffer=max(160, int(rate * 0.02)),
                        stream_callback=self._make_callback(source, gen),
                    )
                    try:
                        if hasattr(stream, "is_active") and not stream.is_active():
                            stream.start_stream()
                    except Exception:
                        pass
                except Exception:
                    continue
                self._formats[(source, gen)] = (rate, channels)
                self._streams[source] = OpenStream(source, spec, channels, rate, stream, gen, self._clock())
                self.device_names[source] = spec.name.replace(" [Loopback]", "")
                return True
        self.device_names[source] = "not found"
        self._streams.pop(source, None)
        return False

    def _make_callback(self, source: str, gen: int):
        q = self._raw_queues[source]
        pyaudio = self._pa()
        cont = getattr(pyaudio, "paContinue", 0)

        def callback(in_data, frame_count, time_info, status):
            if gen == self._generation and self._running:
                q.put((gen, in_data, self._clock()))
                st = self._streams.get(source)
                if st is not None and st.generation == gen:
                    st.last_callback = self._clock()
                    st.callbacks += 1
            return (None, cont)

        return callback

    def _close_all(self):
        with self._lock:
            for source, st in list(self._streams.items()):
                try:
                    if hasattr(st.stream, "is_stopped") and not st.stream.is_stopped():
                        st.stream.stop_stream()
                except Exception:
                    pass
                try:
                    st.stream.close()
                except Exception:
                    pass
            self._streams.clear()
            if self._p is not None:
                try:
                    self._p.terminate()
                except Exception:
                    pass
                self._p = None
            # tell processors to flush their phrases
            for source in SOURCES:
                self._raw_queues[source].put(("flush", None, self._clock()))

    # -- processing threads -----------------------------------------------------
    def _process_loop(self, source: str):
        q = self._raw_queues[source]
        current_gen = None
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    break
                continue
            if item is None:
                break
            gen, data, ts = item
            seg = self._segmenters[source]
            if gen == "flush":
                if seg is not None:
                    for s in seg.flush(ts):
                        self._safe_emit(s)
                self._levels[source] = 0.0
                continue
            if gen != current_gen or seg is None:
                if seg is not None:
                    for s in seg.flush(ts):
                        self._safe_emit(s)
                fmt = self._formats.get((source, gen))
                if fmt is None:
                    continue
                rate, channels = fmt
                seg = self._segmenters[source] = self._make_segmenter(source, rate)
                current_gen = gen
            rate, channels = self._formats.get((source, gen), (seg.sample_rate, 1))
            chunk = pcm16_to_mono_float(data, channels)
            energy = rms_int16(chunk)
            self._levels[source] = max(0.0, min(1.0, energy / 4000.0))
            for s in seg.feed(chunk, ts):
                self._safe_emit(s)

    def _safe_emit(self, segment: Segment):
        try:
            self.on_segment(segment)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[AudioEngine] segment callback error: {exc}")

    # -- watchdog ---------------------------------------------------------------
    def _watchdog_loop(self):
        poll = float(self.config.get("device_poll_interval", 1.0))
        debounce = float(self.config.get("device_change_debounce", 0.8))
        stall = float(self.config.get("mic_stall_timeout", 4.0))
        while self._running:
            self._sleep(poll)
            if not self._running:
                break
            try:
                self.watchdog_tick(debounce=debounce, stall_timeout=stall)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[AudioEngine] watchdog error: {exc}")

    def watchdog_tick(self, debounce: float = 0.8, stall_timeout: float = 4.0) -> Optional[str]:
        """One watchdog iteration.  Returns the reconnect reason if it reconnected."""
        now = self._clock()
        reason = None

        # 1) Windows default device changes
        try:
            snap = self._probe.snapshot()
        except Exception:
            snap = None
        if snap is not None:
            if self._last_snapshot is None:
                self._last_snapshot = snap
            elif snap != self._last_snapshot:
                if self._pending_snapshot != snap:
                    self._pending_snapshot = snap
                    self._pending_since = now
                elif now - self._pending_since >= debounce:
                    self._last_snapshot = snap
                    self._pending_snapshot = None
                    reason = "default device changed"
            else:
                self._pending_snapshot = None

        # 2) stream health
        if reason is None:
            with self._lock:
                for source, st in list(self._streams.items()):
                    active = True
                    try:
                        active = bool(st.stream.is_active())
                    except Exception:
                        active = False
                    if not active:
                        reason = f"{source} stream stopped"
                        break
                    if source == "You" and now - st.last_callback > stall_timeout:
                        reason = "microphone stalled"
                        break

        # 3) missing sources → periodic retry with backoff
        if reason is None:
            with self._lock:
                missing = [s for s in SOURCES if self.enabled_sources.get(s, True) and s not in self._streams]
            if missing and now - self._last_open_attempt >= self._retry_delay:
                reason = "retry " + "/".join(missing)

        if reason:
            self.reconnect_count += 1
            self.on_status("info", f"🔄 Reconnecting audio: {reason}")
            self.reconnect(reason)
        return reason


# --------------------------------------------------------------------------- #
# Manual test:  python audio_engine.py   (prints devices, then live segments)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import pyaudiowpatch as pyaudio  # type: ignore

    p = pyaudio.PyAudio()
    print("Input devices / loopbacks visible to PortAudio:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"  [{i}] {info['name']}  rate={int(info['defaultSampleRate'])}  ch={info['maxInputChannels']}  loopback={info.get('isLoopbackDevice')}")
    p.terminate()

    def on_seg(seg: Segment):
        print(f"{seg.source:8s} {'FINAL' if seg.is_final else 'interim'} {seg.duration:4.1f}s  phrase={seg.phrase_id}")

    eng = AudioEngine({"transcribe_mic": True}, on_segment=on_seg, on_status=lambda lvl, m: print(f"[{lvl}] {m}"))
    eng.start()
    print("Listening… change your default device to test auto-switching. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            lv = eng.levels()
            print(f"  levels: mic={lv['You']:.2f} spk={lv['Speaker']:.2f}   {eng.status_text()}", end="\r")
    except KeyboardInterrupt:
        eng.stop()
