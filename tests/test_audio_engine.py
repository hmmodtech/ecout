import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_engine import (AudioEngine, PhraseSegmenter, pcm16_to_mono_float,  # noqa: E402
                          resample_to_16k, resolve_loopback_candidates, resolve_mic_candidates, rms_int16)
from fake_audio import FakeClock, FakeProbe, FakePyAudioModule, wait_until  # noqa: E402


# --------------------------------------------------------------------------- #
# DSP
# --------------------------------------------------------------------------- #
def tone(seconds, rate, freq=440.0, amp=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pcm_to_mono_and_rms():
    stereo = np.array([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
    mono = pcm16_to_mono_float(stereo, 2)
    assert mono.shape == (2,)
    assert np.allclose(mono, [0.0, 0.0])
    x = tone(0.1, 16000, amp=0.5)
    assert 0.5 * 32768 / np.sqrt(2) * 0.95 < rms_int16(x) < 0.5 * 32768 / np.sqrt(2) * 1.05


@pytest.mark.parametrize("rate", [16000, 32000, 44100, 48000, 96000])
def test_resample_preserves_duration_and_pitch(rate):
    x = tone(1.0, rate, freq=440.0)
    y = resample_to_16k(x, rate)
    assert abs(len(y) - 16000) <= 2
    # dominant frequency must still be 440 Hz
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(freqs[np.argmax(spectrum)] - 440.0) < 5.0


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def feed(seg: PhraseSegmenter, audio: np.ndarray, rate: int, clock: FakeClock, chunk_ms=20):
    out = []
    n = int(rate * chunk_ms / 1000)
    for i in range(0, len(audio), n):
        chunk = audio[i:i + n]
        clock.sleep(len(chunk) / rate)
        out.extend(seg.feed(chunk, clock()))
    return out


def test_segmenter_emits_final_after_pause():
    rate = 48000
    clock = FakeClock()
    seg = PhraseSegmenter(rate, energy_threshold=300, dynamic_energy=False, pause_threshold=0.5,
                          non_speaking_duration=0.2, interim_interval=0)
    audio = np.concatenate([np.zeros(int(0.5 * rate), np.float32), tone(1.5, rate), np.zeros(int(1.0 * rate), np.float32)])
    segs = feed(seg, audio, rate, clock)
    assert len(segs) == 1
    s = segs[0]
    assert s.is_final and s.source == "Speaker" and s.rate == 16000
    # 1.5 s speech + ~0.2 s pre-roll + ~0.2 s tail
    assert 1.6 <= s.duration <= 2.1
    assert abs((s.end_time - s.start_time) - s.duration) < 0.05
    assert s.audio.dtype == np.float32


def test_segmenter_ignores_short_clicks():
    rate = 16000
    clock = FakeClock()
    seg = PhraseSegmenter(rate, dynamic_energy=False, phrase_threshold=0.3, interim_interval=0)
    audio = np.concatenate([np.zeros(8000, np.float32), tone(0.1, rate), np.zeros(16000, np.float32)])
    assert feed(seg, audio, rate, clock) == []


def test_segmenter_hard_cut_and_interims_for_long_speech():
    rate = 16000
    clock = FakeClock()
    seg = PhraseSegmenter(rate, dynamic_energy=False, phrase_time_limit=4.0, interim_interval=1.5)
    audio = np.concatenate([tone(9.0, rate), np.zeros(16000, np.float32)])
    segs = feed(seg, audio, rate, clock)
    finals = [s for s in segs if s.is_final]
    interims = [s for s in segs if not s.is_final]
    assert len(finals) == 3                       # 4 s + 4 s + remaining 1 s
    assert all(f.duration <= 4.05 for f in finals)
    assert len(interims) >= 3
    # interims belong to the phrase that is later finalised
    assert {i.phrase_id for i in interims} <= {f.phrase_id for f in finals}
    # phrases are contiguous
    assert finals[0].phrase_id < finals[1].phrase_id < finals[2].phrase_id


def test_segmenter_dynamic_threshold_has_floor():
    rate = 16000
    clock = FakeClock()
    seg = PhraseSegmenter(rate, energy_threshold=300, dynamic_energy=True, min_energy_threshold=120)
    feed(seg, np.zeros(rate * 5, np.float32), rate, clock)
    assert seg.threshold == pytest.approx(120, abs=1)


def test_segmenter_flush_returns_pending_phrase():
    rate = 16000
    clock = FakeClock()
    seg = PhraseSegmenter(rate, dynamic_energy=False, interim_interval=0)
    feed(seg, tone(1.0, rate), rate, clock)
    segs = seg.flush(clock())
    assert len(segs) == 1 and segs[0].is_final and 0.9 < segs[0].duration < 1.3


# --------------------------------------------------------------------------- #
# device resolution
# --------------------------------------------------------------------------- #
def make_module():
    m = FakePyAudioModule()
    mic = m.add_device("Microphone Array (Realtek)", 2, 44100)
    spk, _ = m.add_output_with_loopback("Speakers (Realtek)", 48000)
    m.set_defaults(mic, spk)
    return m


def test_resolve_candidates_prefer_named_devices():
    m = make_module()
    hs_mic = m.add_device("Headset Microphone (Jabra)", 1, 16000)
    hs_out, hs_lb = m.add_output_with_loopback("Headset Earphone (Jabra)", 48000)
    p = m.PyAudio()
    mics = resolve_mic_candidates(p, m, preferred_name="Headset Microphone (Jabra)")
    assert mics[0].index == hs_mic
    assert mics[1].name.startswith("Microphone Array")
    lbs = resolve_loopback_candidates(p, m, preferred_output_name="Headset Earphone (Jabra)")
    assert lbs[0].index == hs_lb and lbs[0].kind == "loopback"
    lbs = resolve_loopback_candidates(p, m)
    assert lbs[0].name.startswith("Speakers (Realtek)")


# --------------------------------------------------------------------------- #
# engine with fake backend
# --------------------------------------------------------------------------- #
def make_engine(module, probe=None, config=None):
    clock = FakeClock()
    segments = []
    statuses = []
    eng = AudioEngine(config or {"pause_threshold": 0.5, "interim_interval": 0, "dynamic_energy": False},
                      on_segment=segments.append, on_status=lambda l, m: statuses.append((l, m)),
                      pyaudio_module=module, probe=probe or FakeProbe(module), sleep=clock.sleep, clock=clock)
    eng._watchdog_loop = lambda: None  # drive the watchdog manually
    return eng, clock, segments, statuses


def test_engine_opens_both_streams_and_emits_segments():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        assert len(m.streams_for("Microphone Array")) == 1
        assert len(m.streams_for("Speakers (Realtek) [Loopback]")) == 1
        assert eng.device_names["You"].startswith("Microphone Array")
        assert eng.device_names["Speaker"] == "Speakers (Realtek)"
        assert m.instances == 1

        lb = m.streams_for("Speakers (Realtek) [Loopback]")[0]
        assert lb.channels == 2 and lb.rate == 48000
        # simulate the interviewer speaking for 1.2 s then silence
        lb.push(0.3, 0.0)
        lb.push(1.2, 0.3)
        lb.push(1.0, 0.0)
        assert wait_until(lambda: any(s.source == "Speaker" and s.is_final for s in segments))
        seg = segments[-1]
        assert 1.2 <= seg.duration <= 1.8 and seg.rate == 16000
        assert eng.levels()["Speaker"] >= 0.0
    finally:
        eng.stop()


def test_engine_switches_to_new_default_device():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        old_mic = m.streams_for("Microphone Array")[0]
        old_lb = m.streams_for("Speakers (Realtek) [Loopback]")[0]

        # user connects a headset and Windows makes it default
        hs_mic = m.add_device("Headset Microphone (Jabra)", 1, 16000)
        hs_out, _ = m.add_output_with_loopback("Headset Earphone (Jabra)", 48000)
        m.set_defaults(hs_mic, hs_out)

        assert eng.watchdog_tick(debounce=0.8) is None      # first observation → pending (debounce)
        clock.sleep(1.0)
        reason = eng.watchdog_tick(debounce=0.8)
        assert reason == "default device changed"

        assert old_mic.closed and old_lb.closed
        assert len(m.streams_for("Headset Microphone")) == 1
        assert len(m.streams_for("Headset Earphone (Jabra) [Loopback]")) == 1
        assert eng.device_names["You"].startswith("Headset Microphone")
        assert eng.device_names["Speaker"] == "Headset Earphone (Jabra)"
        assert m.instances == 1                              # old PortAudio instance terminated
        assert eng.reconnect_count == 1

        # audio keeps flowing through the *same* pipeline after the switch
        new_lb = m.streams_for("Headset Earphone (Jabra) [Loopback]")[0]
        new_lb.push(1.0, 0.3)
        new_lb.push(1.0, 0.0)
        assert wait_until(lambda: any(s.source == "Speaker" and s.is_final for s in segments))
        # no further reconnects while nothing changes
        clock.sleep(2)
        assert eng.watchdog_tick() is None
    finally:
        eng.stop()


def test_engine_reconnects_when_stream_dies():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        mic = m.streams_for("Microphone Array")[0]
        mic.active = False                                    # device yanked
        assert eng.watchdog_tick() == "You stream stopped"
        assert mic.closed
        assert len(m.streams_for("Microphone Array")) == 1    # re-opened
        assert eng.reconnect_count == 1
    finally:
        eng.stop()


def test_engine_detects_stalled_microphone():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        clock.sleep(10)
        assert eng.watchdog_tick(stall_timeout=4.0) == "microphone stalled"
    finally:
        eng.stop()


def test_engine_retries_when_no_microphone_available():
    m = FakePyAudioModule()
    spk, _ = m.add_output_with_loopback("Speakers (Realtek)", 48000)
    m.set_defaults(-1, spk)
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        assert "You" not in eng._streams and "Speaker" in eng._streams
        assert eng.device_names["You"] == "not found"
        assert any(l == "warning" for l, _ in statuses)
        # microphone appears later
        mic = m.add_device("USB Microphone", 1, 48000)
        m.set_defaults(mic, spk)
        clock.sleep(1.5)
        assert eng.watchdog_tick(debounce=0.8) is None       # change observed, debounce running
        clock.sleep(1.5)
        assert eng.watchdog_tick(debounce=0.8) is not None   # reconnect (device change / retry)
        assert wait_until(lambda: "You" in eng._streams)
        assert eng.device_names["You"] == "USB Microphone"
    finally:
        eng.stop()


def test_engine_falls_back_to_supported_open_parameters():
    m = FakePyAudioModule()
    mic = m.add_device("Picky Mic", 2, 48000, allowed_rates=[48000])
    spk, _ = m.add_output_with_loopback("Speakers", 44100)
    m.set_defaults(mic, spk)
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        st = eng._streams["You"]
        assert st.rate == 48000
        lb = eng._streams["Speaker"]
        assert lb.rate == 44100 and lb.channels == 2
    finally:
        eng.stop()


def test_engine_without_probe_still_handles_stream_errors():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m, probe=FakeProbe(m, available=False))
    eng.start()
    try:
        assert eng.watchdog_tick() is None
        m.streams_for("Speakers (Realtek) [Loopback]")[0].active = False
        assert eng.watchdog_tick() == "Speaker stream stopped"
    finally:
        eng.stop()


def test_manual_reconnect_flushes_pending_phrase():
    m = make_module()
    eng, clock, segments, statuses = make_engine(m)
    eng.start()
    try:
        lb = m.streams_for("Speakers (Realtek) [Loopback]")[0]
        lb.push(1.0, 0.3)                       # speaking, no pause yet
        eng.reconnect("manual")
        assert wait_until(lambda: any(s.is_final for s in segments))
        assert 0.9 <= segments[-1].duration <= 1.3
    finally:
        eng.stop()
