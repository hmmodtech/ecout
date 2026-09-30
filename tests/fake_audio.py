"""A fake ``pyaudiowpatch`` module + default-device probe for tests.

Mimics the one PortAudio behaviour that matters for auto-switching: the
device list is frozen when ``PyAudio()`` is instantiated, so a change of the
Windows default device only becomes visible after terminate()+re-init.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np


class FakeStream:
    def __init__(self, module, device, channels, rate, frames, callback):
        self.module = module
        self.device = device
        self.channels = channels
        self.rate = rate
        self.frames = frames
        self.callback = callback
        self.active = True
        self.stopped = False
        self.closed = False

    # PyAudio API
    def is_active(self):
        return self.active and not self.stopped and not self.closed

    def is_stopped(self):
        return self.stopped

    def start_stream(self):
        self.stopped = False

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True

    # test helpers
    def push(self, seconds: float, amplitude: float = 0.0, freq: float = 440.0, chunk_frames: Optional[int] = None):
        """Deliver ``seconds`` of audio to the callback (tone or silence)."""
        if self.closed:
            return
        n = chunk_frames or self.frames
        total = int(seconds * self.rate)
        t0 = 0
        while t0 < total and not self.closed and not self.stopped:
            count = min(n, total - t0)
            t = (np.arange(count) + t0) / self.rate
            mono = amplitude * np.sin(2 * np.pi * freq * t)
            pcm = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
            if self.channels > 1:
                pcm = np.repeat(pcm[:, None], self.channels, axis=1).reshape(-1)
            self.callback(pcm.tobytes(), count, {}, 0)
            t0 += count


class FakePyAudioInstance:
    def __init__(self, module):
        self.module = module
        module.instances += 1
        # frozen snapshot – this is the PortAudio behaviour we emulate
        self.devices: List[dict] = [dict(d) for d in module.devices]
        self.defaults = dict(module.defaults)
        self.terminated = False

    def get_host_api_info_by_type(self, api_type):
        assert api_type == self.module.paWASAPI
        return {"index": 0, "name": "Windows WASAPI", "deviceCount": len(self.devices),
                "defaultInputDevice": self.defaults["input"], "defaultOutputDevice": self.defaults["output"]}

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, idx):
        return self.devices[idx]

    def get_default_input_device_info(self):
        if self.defaults["input"] < 0:
            raise OSError("no default input")
        return self.devices[self.defaults["input"]]

    def get_loopback_device_info_generator(self):
        for d in self.devices:
            if d.get("isLoopbackDevice"):
                yield d

    def get_default_wasapi_loopback(self):
        out = self.devices[self.defaults["output"]]
        for d in self.devices:
            if d.get("isLoopbackDevice") and d["name"].startswith(out["name"]):
                return d
        raise OSError("no loopback")

    def open(self, format, channels, rate, input, input_device_index, frames_per_buffer, stream_callback, **kw):
        dev = self.devices[input_device_index]
        if dev.get("broken"):
            raise OSError("device unavailable")
        if channels > dev["maxInputChannels"] or channels <= 0:
            raise OSError("invalid channel count")
        allowed = dev.get("allowed_rates")
        if allowed and rate not in allowed:
            raise OSError("invalid sample rate")
        stream = FakeStream(self.module, dev, channels, rate, frames_per_buffer, stream_callback)
        self.module.open_streams.append(stream)
        return stream

    def terminate(self):
        self.terminated = True
        self.module.instances -= 1


class FakePyAudioModule:
    paInt16 = 8
    paWASAPI = 13
    paContinue = 0

    def __init__(self):
        self.devices: List[dict] = []
        self.defaults = {"input": -1, "output": -1}
        self.open_streams: List[FakeStream] = []
        self.instances = 0
        self.endpoints = {}

    def PyAudio(self):
        return FakePyAudioInstance(self)

    # -- helpers to build device setups -------------------------------------------
    def add_device(self, name, max_in=0, rate=48000, loopback=False, broken=False, allowed_rates=None):
        info = {"index": len(self.devices), "name": name, "maxInputChannels": max_in, "maxOutputChannels": 0 if max_in else 2,
                "defaultSampleRate": rate, "hostApi": 0, "isLoopbackDevice": loopback, "broken": broken}
        if allowed_rates:
            info["allowed_rates"] = allowed_rates
        self.devices.append(info)
        return info["index"]

    def add_output_with_loopback(self, name, rate=48000):
        out_idx = self.add_device(name, 0, rate)
        lb_idx = self.add_device(f"{name} [Loopback]", 2, rate, loopback=True)
        return out_idx, lb_idx

    def set_defaults(self, mic_idx, out_idx, mic_id=None, out_id=None):
        self.defaults = {"input": mic_idx, "output": out_idx}
        mic = self.devices[mic_idx]["name"] if mic_idx >= 0 else None
        out = self.devices[out_idx]["name"] if out_idx >= 0 else None
        self.endpoints = {
            "render_mm": (out_id or f"id-{out}", out), "render_comm": (out_id or f"id-{out}", out),
            "capture_mm": (mic_id or f"id-{mic}", mic), "capture_comm": (mic_id or f"id-{mic}", mic),
        }

    def streams_for(self, name_prefix):
        return [s for s in self.open_streams if s.device["name"].startswith(name_prefix) and not s.closed]


class FakeProbe:
    def __init__(self, module: FakePyAudioModule, available=True):
        self.module = module
        self.available = available
        self.backend = "fake"

    def snapshot(self):
        if not self.available:
            return None
        return dict(self.module.endpoints)


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def wait_until(predicate, timeout=3.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
