"""Headless tests for the wiring in main.py (Tk replaced by a fake)."""
import os
import sys
import types


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_audio import wait_until  # noqa: E402


# --------------------------------------------------------------------------- #
# a tiny fake customtkinter
# --------------------------------------------------------------------------- #
class FakeWidget:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.text = ""
        self._value = kwargs.get("variable")
        self._selected = False

    def __getattr__(self, name):
        # pack/grid/bind/... → no-op
        return lambda *a, **k: None

    def configure(self, **kw):
        self.kwargs.update(kw)

    # textbox API
    def delete(self, *a):
        self.text = ""

    def insert(self, index, text, *a):
        if index in ("0.0", "1.0"):
            self.text = text + self.text
        else:
            self.text += text

    def get(self, *a):
        if self._value is not None:
            return self._value.get()
        return self._selected

    def set(self, v):
        self._value_set = v

    def select(self):
        self._selected = True

    def deselect(self):
        self._selected = False


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, v):
        self.value = v


class FakeRoot(FakeWidget):
    def __init__(self):
        super().__init__()
        self.pending = []

    def after(self, ms, fn=None, *args):
        if fn is not None:
            self.pending.append((fn, args))

    def pump(self):
        # process what is queued now (the UI loop re-schedules itself)
        items, self.pending = self.pending, []
        for fn, args in items:
            fn(*args)

    def winfo_id(self):
        return 0


def install_fake_gui(monkeypatch):
    ctk = types.ModuleType("customtkinter")
    for name in ["CTkFrame", "CTkButton", "CTkLabel", "CTkTextbox", "CTkProgressBar", "CTkSegmentedButton",
                 "CTkSwitch", "CTkEntry", "CTkScrollableFrame", "CTkOptionMenu", "CTkToplevel"]:
        setattr(ctk, name, FakeWidget)
    ctk.CTk = FakeRoot
    ctk.StringVar = FakeVar
    ctk.BooleanVar = FakeVar
    ctk.set_appearance_mode = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "customtkinter", ctk)
    tk = types.ModuleType("tkinter")
    tk.filedialog = types.SimpleNamespace(askopenfilename=lambda **k: "")
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", tk.filedialog)
    monkeypatch.setitem(sys.modules, "keyboard", None)


class FakeASR:
    name = "fake-asr"

    def __init__(self):
        self.vocab = ""

    def set_vocabulary(self, words):
        self.vocab = words

    def transcribe(self, audio):
        return ""


def make_app(monkeypatch, tmp_path):
    install_fake_gui(monkeypatch)
    monkeypatch.chdir(tmp_path)
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "qa_bank.txt").write_text(
        "Q: Tell me about yourself.\nA: I am a security professional with eight years in the field.\n"
        "Q: How do you handle a hostile checkpoint?\nA: Stay calm, keep hands visible, comply and report.\n",
        encoding="utf-8")
    sys.modules.pop("main", None)
    import main

    # no real audio / ASR in tests
    class NoAudio:
        def __init__(self, *a, **k):
            self.device_names = {"You": "Fake Mic", "Speaker": "Fake Speakers"}
            self.enabled_sources = {"You": True, "Speaker": True}

        def start(self):
            pass

        def stop(self):
            pass

        def reconnect(self, reason="manual"):
            self.reason = reason

        def levels(self):
            return {"You": 0.0, "Speaker": 0.0}

        def status_text(self):
            return "🎤 Fake Mic | 🔊 Fake Speakers"

    monkeypatch.setattr(main, "AudioEngine", NoAudio)
    fake_models = types.ModuleType("TranscriberModels")
    fake_models.get_model = lambda use_api=False, config=None: FakeASR()
    fake_models.clean_transcript = lambda t: t
    monkeypatch.setitem(sys.modules, "TranscriberModels", fake_models)

    config = main.DEFAULT_CONFIG.copy()
    config.update({"gemini_api_key": "", "semantic_search": False, "auto_answer": True})
    root = main.ctk.CTk()
    app = main.AbujamiApp(root, config, use_api=False)
    assert wait_until(lambda: app.ready, timeout=10)
    root.pump()
    return main, app, root


def speaker_phrase(app, pid, text, final=True, start=0.0):
    from AudioTranscriber import TranscriptEntry
    import time

    now = time.time()
    entry = TranscriptEntry(pid, "Speaker", text, now - 1 + start, now + start, final, pid)
    with app.transcriber._lock:
        app.transcriber._entries[pid] = entry
        app.transcriber._bump()
    app._on_transcript_update(entry)


def test_app_boots_headless_and_loads_knowledge(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    assert app.kb.stats.qa_pairs["QA"] == 2
    assert app.answer_engine is not None and app.transcriber is not None
    assert "Auto-answer is ON" in app.answer_box.text
    assert app.asr_model.vocab is not None


def test_auto_answer_from_qa_bank_without_llm(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    speaker_phrase(app, 1, "Thanks for joining. So, tell me about yourself?")
    assert wait_for_text(root, app, "QA Bank")
    assert "eight years" in app.answer_box.text
    assert "tell me about yourself" in app.last_answered_key


def test_same_question_is_not_answered_twice_but_new_one_is(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    speaker_phrase(app, 1, "Tell me about yourself?")
    assert wait_for_text(root, app, "eight years")
    gen = app.answer_generation
    speaker_phrase(app, 1, "Tell me about yourself?")            # duplicate (e.g. interim → final)
    assert app.answer_generation == gen
    speaker_phrase(app, 2, "And how do you handle a hostile checkpoint?", start=5)
    assert wait_until(lambda: app.answer_generation == gen + 1, timeout=5)
    assert wait_for_text(root, app, "Stay calm")


def test_statements_do_not_trigger_answers(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    speaker_phrase(app, 1, "We are a humanitarian organisation working in twelve countries.")
    speaker_phrase(app, 2, "The role is based in Juba.", start=2)
    assert app.answer_generation == 0


def test_manual_trigger_uses_last_phrase_when_no_question(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    app.auto_answer = False
    speaker_phrase(app, 1, "Hostile checkpoint handling.")
    app.trigger_answer()
    assert wait_for_text(root, app, "Hostile checkpoint handling")
    assert app.answer_view["question"] == "Hostile checkpoint handling."


def test_meeting_mode_builds_paragraphs(monkeypatch, tmp_path):
    main, app, root = make_app(monkeypatch, tmp_path)
    app.toggle_mode("Meeting")
    app.handle_main_action()  # start recording
    assert app.is_recording_meeting
    from AudioTranscriber import TranscriptEntry
    import time

    now = time.time()
    with app.transcriber._lock:
        app.transcriber._entries[1] = TranscriptEntry(1, "Speaker", "Welcome everyone to the call.", now, now + 1, True, 1)
        app.transcriber._entries[2] = TranscriptEntry(2, "Speaker", "Let us start with the agenda.", now + 1, now + 2, True, 2)
        app.transcriber._entries[3] = TranscriptEntry(3, "You", "Thanks, happy to be here.", now + 2, now + 3, True, 3)
        app.transcriber._bump()
    app._ui_tick()
    assert len(app.meeting_paragraphs) == 2
    assert app.meeting_paragraphs[0]["speaker"] == "[Remote / Teams]"
    assert "agenda" in app.meeting_buffer or "agenda" in app.meeting_paragraphs[0]["text"] or "agenda" in app.transcript_box.text
    assert app.meeting_paragraphs[1]["speaker"] == "[Candidate / You]"


def test_export_meeting_to_word(monkeypatch, tmp_path):
    install_fake_gui(monkeypatch)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("main", None)
    import main

    path = main.export_meeting_to_word([{"speaker": "[Remote / Teams]", "time": "10:00:00", "text": "Hello"}])
    assert path and os.path.exists(path)


def wait_for_text(root, app, needle, timeout=5.0):
    """Pump the fake Tk loop until the answer box contains ``needle``."""
    def check():
        root.pump()
        return needle in app.answer_box.text
    return wait_until(check, timeout=timeout)
