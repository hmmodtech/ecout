import custom_speech_recognition as sr
import pyaudiowpatch as pyaudio
from datetime import datetime

class BaseRecorder:
    def __init__(self, source=None):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 300
        self.recorder.dynamic_energy_threshold = True
        self.recorder.dynamic_energy_adjustment_damping = 0.15
        self.recorder.dynamic_energy_ratio = 1.5
        self.recorder.pause_threshold = 0.8
        self.source = source

    def adjust_for_noise(self, device_name, msg):
        try:
            if self.source:
                with self.source:
                    self.recorder.adjust_for_ambient_noise(self.source, duration=0.5)
        except Exception:
            pass

    def record_into_queue(self, audio_queue):
        def record_callback(recognizer, audio):
            data = audio.get_raw_data()
            audio_queue.put((data, datetime.utcnow()))

        if not self.source:
            return None

        try:
            return self.recorder.listen_in_background(
                self.source, 
                record_callback, 
                phrase_time_limit=3
            )
        except Exception as e:
            print(f"[Audio Error] {e}")
            return None

class DefaultMicRecorder(BaseRecorder):
    def __init__(self):
        source = None
        p = pyaudio.PyAudio()
        try:
            default_input = p.get_default_input_device_info()
            device_index = default_input["index"]
            sample_rate = int(default_input["defaultSampleRate"])
            source = sr.Microphone(device_index=device_index, sample_rate=sample_rate)
        except Exception:
            source = sr.Microphone()
        finally:
            p.terminate()

        super().__init__(source=source)
        self.adjust_for_noise("Default Mic", "Calibrating Mic...")

class DefaultSpeakerRecorder(BaseRecorder):
    def __init__(self):
        source = None
        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            if not default_speakers["isLoopbackDevice"]:
                found_loopback = False
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        found_loopback = True
                        break
                if not found_loopback:
                    default_speakers = p.get_default_wasapi_loopback()

            source = sr.Microphone(
                speaker=True,
                device_index=default_speakers["index"],
                sample_rate=int(default_speakers["defaultSampleRate"])
            )
        except Exception:
            try:
                default_speakers = p.get_default_wasapi_loopback()
                source = sr.Microphone(
                    speaker=True,
                    device_index=default_speakers["index"],
                    sample_rate=int(default_speakers["defaultSampleRate"])
                )
            except Exception:
                source = None
        finally:
            p.terminate()

        super().__init__(source=source)
        self.adjust_for_noise("Default Speaker", "Calibrating Speaker...")