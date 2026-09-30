import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AudioTranscriber import AudioTranscriber  # noqa: E402
from TranscriberModels import clean_transcript, float_to_wav_bytes, load_wav_as_float  # noqa: E402
from audio_engine import Segment  # noqa: E402
from fake_audio import wait_until  # noqa: E402


class FakeModel:
    """Returns a text encoded in the audio's first sample; optionally slow."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.gate.set()

    def transcribe(self, audio):
        self.gate.wait()
        time.sleep(self.delay)
        code = int(round(float(audio[0]) * 1000))
        with self.lock:
            self.calls.append(code)
        return {0: "", 1: "Tell me about yourself?", 2: "I am a security officer", 3: "you",
                4: "partial question", 5: "full question about checkpoints?"}.get(code, f"text {code}")


def seg(code, source="Speaker", final=True, pid=None, start=0.0, dur=1.0):
    audio = np.zeros(16000, dtype=np.float32)
    audio[0] = code / 1000.0
    pid = pid or code
    return Segment(source, audio, start, start + dur, final, pid)


def test_clean_transcript_filters_hallucinations():
    assert clean_transcript(" Thank you. ") == ""
    assert clean_transcript("you") == ""
    assert clean_transcript("...") == ""
    assert clean_transcript("you you you you") == ""
    assert clean_transcript("  Tell   me about yourself. ") == "Tell me about yourself."


def test_wav_roundtrip(tmp_path):
    audio = (0.25 * np.sin(np.arange(16000) / 10.0)).astype(np.float32)
    path = tmp_path / "x.wav"
    path.write_bytes(float_to_wav_bytes(audio))
    back = load_wav_as_float(str(path))
    assert back.shape == audio.shape
    assert np.allclose(back, audio, atol=1e-3)


def test_transcriber_orders_and_renders():
    model = FakeModel()
    tr = AudioTranscriber(model, {"asr_debug": False}, workers=1)
    tr.start()
    try:
        tr.submit(seg(2, "You", start=1.0))
        tr.submit(seg(1, "Speaker", start=2.0))
        assert wait_until(lambda: len(tr.get_entries()) == 2)
        text = tr.get_transcript(newest_first=True)
        assert text.startswith("Speaker: [Tell me about yourself?]")
        assert "You: [I am a security officer]" in text
        recent = tr.recent_phrases("Speaker", seconds=60, now=10.0)
        assert recent == [(1, "Tell me about yourself?", 3.0)]
        assert tr.version >= 2
    finally:
        tr.stop()


def test_speaker_finals_are_prioritised_over_mic_and_interims():
    model = FakeModel()
    tr = AudioTranscriber(model, workers=1)
    # queue everything before the worker starts so ordering is deterministic
    tr.submit(seg(2, "You", final=True, pid=10))
    tr.submit(seg(4, "Speaker", final=False, pid=11))
    tr.submit(seg(1, "Speaker", final=True, pid=12))
    tr.start()
    try:
        assert wait_until(lambda: len(model.calls) == 3)
        assert model.calls == [1, 2, 4]  # speaker final, mic final, then the interim
    finally:
        tr.stop()


def test_interims_are_coalesced_and_replaced_by_final():
    model = FakeModel()
    model.gate.clear()
    tr = AudioTranscriber(model, workers=1)
    tr.start()
    try:
        tr.submit(seg(4, "Speaker", final=False, pid=20, start=0, dur=2))
        tr.submit(seg(4, "Speaker", final=False, pid=20, start=0, dur=3))   # newer interim, same phrase
        time.sleep(0.05)
        model.gate.set()
        assert wait_until(lambda: len(model.calls) >= 1)
        time.sleep(0.1)
        assert model.calls == [4]                          # only ONE interim decode
        assert not tr.get_entries()[0].is_final
        assert tr.get_transcript().startswith("Speaker: [partial question …]")

        tr.submit(seg(5, "Speaker", final=True, pid=20, start=0, dur=4))
        assert wait_until(lambda: tr.get_entries() and tr.get_entries()[0].is_final)
        entries = tr.get_entries()
        assert len(entries) == 1 and entries[0].text == "full question about checkpoints?"
    finally:
        tr.stop()


def test_hallucinated_final_removes_interim_entry():
    model = FakeModel()
    tr = AudioTranscriber(model, workers=1)
    tr.start()
    try:
        tr.submit(seg(4, "Speaker", final=False, pid=30))
        assert wait_until(lambda: len(tr.get_entries()) == 1)
        tr.submit(seg(3, "Speaker", final=True, pid=30))  # "you" → filtered → empty
        assert wait_until(lambda: len(tr.get_entries()) == 0)
    finally:
        tr.stop()


def test_on_update_callback_and_clear():
    model = FakeModel()
    updates = []
    tr = AudioTranscriber(model, on_update=updates.append, workers=2)
    tr.start()
    try:
        for i in range(5):
            tr.submit(seg(1, "Speaker", pid=100 + i, start=i))
        assert wait_until(lambda: len(updates) == 5)
        assert all(u.who == "Speaker" for u in updates)
        tr.clear_transcript_data()
        assert tr.get_entries() == []
        assert tr.average_latency_ms() >= 0
    finally:
        tr.stop()
