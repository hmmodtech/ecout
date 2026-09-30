"""
Transcription scheduler + transcript store.

Speech *segments* produced by :mod:`audio_engine` are transcribed by a small
worker pool.  Interviewer (Speaker) finals are served first, interim
segments are coalesced (only the newest one per phrase is ever decoded), so
a question is recognised as soon as the interviewer pauses – even while the
candidate's own audio is still being processed.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from audio_engine import Segment
from TranscriberModels import clean_transcript

MAX_PHRASES = 40  # kept in memory / rendered


@dataclass
class TranscriptEntry:
    phrase_id: int
    who: str
    text: str
    start_time: float
    end_time: float
    is_final: bool
    seq: int = 0
    latency_ms: float = 0.0

    def line(self) -> str:
        suffix = "" if self.is_final else " …"
        return f"{self.who}: [{self.text}{suffix}]\n\n"


class AudioTranscriber:
    def __init__(self, model, config: Optional[dict] = None,
                 on_update: Optional[Callable[[TranscriptEntry], None]] = None,
                 workers: int = 2):
        self.audio_model = model
        self.config = config or {}
        self.on_update = on_update or (lambda entry: None)
        self.workers = max(1, int(workers))

        self._jobs: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count(1)
        self._latest_seq: Dict[int, int] = {}          # phrase_id → newest submitted seq
        self._final_seen: set = set()                   # phrase ids that already have a final
        self._entries: Dict[int, TranscriptEntry] = {}  # phrase_id → entry
        self._lock = threading.RLock()
        self._threads: List[threading.Thread] = []
        self._running = False
        self.version = 0
        self.transcript_changed_event = threading.Event()
        self.stats = {"segments": 0, "decoded": 0, "skipped": 0, "total_latency_ms": 0.0}

    # -- lifecycle ---------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"asr-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for _ in self._threads:
            self._jobs.put(((9, 9, 0), 0, None))

    # -- input -------------------------------------------------------------------
    def submit(self, segment: Segment):
        """Called from the audio thread – must be cheap."""
        if segment is None or segment.audio is None or len(segment.audio) == 0:
            return
        seq = next(self._seq)
        with self._lock:
            self._latest_seq[segment.phrase_id] = seq
            if segment.is_final:
                self._final_seen.add(segment.phrase_id)
            self.stats["segments"] += 1
        priority = (0 if segment.is_final else 1, 0 if segment.source == "Speaker" else 1, seq)
        self._jobs.put((priority, seq, segment))

    # -- workers -----------------------------------------------------------------
    def _worker(self):
        while self._running:
            try:
                _prio, seq, segment = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                break
            try:
                self._handle(seq, segment)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[ASR] worker error: {exc}")

    def _handle(self, seq: int, segment: Segment):
        with self._lock:
            newest = self._latest_seq.get(segment.phrase_id, seq)
            if not segment.is_final and (newest != seq or segment.phrase_id in self._final_seen):
                self.stats["skipped"] += 1
                return  # stale interim – a newer decode of this phrase is queued
        t0 = time.perf_counter()
        text = self.audio_model.transcribe(segment.audio) or ""
        latency = (time.perf_counter() - t0) * 1000.0
        text = clean_transcript(text)
        with self._lock:
            self.stats["decoded"] += 1
            self.stats["total_latency_ms"] += latency
            existing = self._entries.get(segment.phrase_id)
            if existing and existing.is_final and not segment.is_final:
                return  # final already displayed
            if not text:
                if existing and not existing.is_final and segment.is_final:
                    # interim showed something but the final decode is empty → drop it
                    self._entries.pop(segment.phrase_id, None)
                    self._bump()
                return
            if existing and not segment.is_final and existing.seq > seq:
                return
            entry = TranscriptEntry(segment.phrase_id, segment.source, text, segment.start_time,
                                    segment.end_time, segment.is_final, seq, latency)
            self._entries[segment.phrase_id] = entry
            self._trim()
            self._bump()
        if self.config.get("asr_debug", False):
            print(f"[ASR] {segment.source:7s} {'final ' if segment.is_final else 'interim'} "
                  f"{segment.duration:4.1f}s audio → {latency:5.0f} ms : {text[:80]}")
        self.on_update(entry)

    def _trim(self):
        if len(self._entries) <= MAX_PHRASES:
            return
        ordered = sorted(self._entries.values(), key=lambda e: e.start_time)
        for old in ordered[: len(ordered) - MAX_PHRASES]:
            self._entries.pop(old.phrase_id, None)
            self._latest_seq.pop(old.phrase_id, None)
            self._final_seen.discard(old.phrase_id)

    def _bump(self):
        self.version += 1
        self.transcript_changed_event.set()

    # -- output ------------------------------------------------------------------
    def get_entries(self, final_only: bool = False, who: Optional[str] = None) -> List[TranscriptEntry]:
        with self._lock:
            items = list(self._entries.values())
        if final_only:
            items = [e for e in items if e.is_final]
        if who:
            items = [e for e in items if e.who == who]
        items.sort(key=lambda e: e.start_time)
        return items

    def get_transcript(self, newest_first: bool = True, limit: int = 12) -> str:
        items = self.get_entries()
        if newest_first:
            items = list(reversed(items))
        return "".join(e.line() for e in items[:limit])

    def recent_phrases(self, who: str = "Speaker", seconds: float = 20.0, now: Optional[float] = None,
                       include_interim: bool = True) -> List[Tuple[int, str, float]]:
        now = time.time() if now is None else now
        out = []
        for e in self.get_entries(who=who):
            if not include_interim and not e.is_final:
                continue
            if now - e.end_time <= seconds:
                out.append((e.phrase_id, e.text, e.end_time))
        return out

    def average_latency_ms(self) -> float:
        with self._lock:
            n = self.stats["decoded"]
            return self.stats["total_latency_ms"] / n if n else 0.0

    def clear_transcript_data(self):
        with self._lock:
            self._entries.clear()
            self._latest_seq.clear()
            self._final_seen.clear()
            # drop queued work
            try:
                while True:
                    self._jobs.get_nowait()
            except queue.Empty:
                pass
            self._bump()
