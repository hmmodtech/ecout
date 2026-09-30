"""
Abujami Assistant – real-time interview copilot / meeting logger.

Pipeline
--------
mic + system audio  →  audio_engine (VAD, auto device switching)
                    →  AudioTranscriber (faster-whisper, speaker-first scheduling)
                    →  question_detector  →  AnswerEngine (Q&A file → HEAT → Gemini)
                    →  UI
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import customtkinter as ctk
from tkinter import filedialog

from AudioTranscriber import AudioTranscriber, TranscriptEntry
from answer_engine import AnswerEngine, AnswerEvent
from audio_engine import AudioEngine, Segment
from knowledge_base import CONTEXT_DIR, KnowledgeBase
from llm_client import GeminiClient, GeminiEmbeddings, resolve_api_key
from question_detector import extract_question, normalize_for_dedupe, question_score
from retrieval import SearchIndex

try:
    import keyboard  # global hotkey (optional)
except Exception:  # pragma: no cover
    keyboard = None

try:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    DOCX_AVAILABLE = True
except ImportError:  # pragma: no cover
    DOCX_AVAILABLE = False

CONFIG_FILE = "config.json"
LOCAL_CONFIG_FILE = "config.local.json"   # git-ignored: secrets live here
SECRET_KEYS = ("gemini_api_key", "openai_api_key")
RECORDINGS_DIR = "recordings"

DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "model_name": "gemini-2.5-flash",
    "embedding_model": "gemini-embedding-001",
    "alpha": 0.82,
    "stealth_mode": True,
    "hotkey": "alt+q",
    "max_tokens": 400,
    "temperature": 0.3,
    "llm_thinking": "minimal",
    "language_mode": "English Only",
    "show_visualizer": True,
    # answering
    "auto_answer": True,
    "auto_answer_min_score": 0.45,
    "answer_window_seconds": 20,
    "qa_high_confidence": 0.62,
    "qa_medium_confidence": 0.40,
    "semantic_search": True,
    "ai_refine_local_answers": False,
    # speech recognition
    "asr_backend": "local",          # local | api
    "asr_api_provider": "openai",    # openai | gemini
    "whisper_model": "base.en",
    "whisper_device": "auto",
    "whisper_compute_type": "auto",
    "asr_beam_size": 1,
    "asr_workers": 2,
    "asr_domain_prompt": True,
    "asr_debug": False,
    "transcribe_mic": True,
    # audio capture / segmentation
    "pause_threshold": 0.5,
    "phrase_time_limit": 8.0,
    "interim_interval": 2.5,
    "energy_threshold": 300,
    "min_energy_threshold": 120,
    "device_role": "auto",           # auto | multimedia | communications
    "device_poll_interval": 1.0,
}

WHISPER_CHOICES = ["tiny.en", "base.en", "small.en", "distil-small.en", "medium.en", "distil-medium.en", "large-v3-turbo"]
DEVICE_ROLE_CHOICES = ["auto", "multimedia", "communications"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """DEFAULT_CONFIG ← config.json ← config.local.json (secrets, git-ignored)."""
    cfg = DEFAULT_CONFIG.copy()
    for path in (CONFIG_FILE, LOCAL_CONFIG_FILE):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg.update(json.load(f))
            except Exception as exc:
                print(f"[Config] could not read {path}: {exc}")
    return cfg


def save_config(cfg: dict):
    """Write settings to config.json and API keys to config.local.json so a
    key can never end up in git again."""
    public = {k: v for k, v in cfg.items() if k not in SECRET_KEYS}
    public.update({k: "" for k in SECRET_KEYS if k in cfg})
    secrets = {k: v for k, v in cfg.items() if k in SECRET_KEYS and v}
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(public, f, indent=4, ensure_ascii=False)
        if secrets or os.path.exists(LOCAL_CONFIG_FILE):
            with open(LOCAL_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(secrets, f, indent=4, ensure_ascii=False)
    except Exception as exc:
        print(f"[Config] could not save: {exc}")


# --------------------------------------------------------------------------- #
# Windows helpers
# --------------------------------------------------------------------------- #
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def set_stealth_mode(hwnd, enable=True) -> bool:
    try:
        affinity = WDA_EXCLUDEFROMCAPTURE if enable else 0
        return ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity) != 0  # type: ignore[attr-defined]
    except Exception:
        return False


def open_path(path: str):
    try:
        os.startfile(os.path.abspath(path))  # type: ignore[attr-defined]
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Word export (meeting mode)
# --------------------------------------------------------------------------- #
def export_meeting_to_word(paragraphs: List[dict]) -> Optional[str]:
    if not paragraphs:
        return None
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if DOCX_AVAILABLE:
        filepath = os.path.join(RECORDINGS_DIR, f"Meeting_{ts}.docx")
        doc = docx.Document()
        title = doc.add_heading("Meeting Record & Transcript", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
        doc.add_paragraph("─" * 45)
        for p in paragraphs:
            head = doc.add_paragraph()
            r_time = head.add_run(f"[{p['time']}] ")
            r_time.font.color.rgb = RGBColor(120, 120, 120)
            r_spk = head.add_run(f"{p['speaker']}:")
            r_spk.bold = True
            r_spk.font.color.rgb = RGBColor(31, 110, 235) if "Teams" in p["speaker"] else RGBColor(35, 134, 54)
            body = doc.add_paragraph(p["text"].strip())
            body.paragraph_format.space_after = Pt(10)
        doc.save(filepath)
    else:
        filepath = os.path.join(RECORDINGS_DIR, f"Meeting_{ts}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(f"[{p['time']}] {p['speaker']}:\n{p['text']}\n\n")
    return filepath


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class AbujamiApp:
    def __init__(self, root: ctk.CTk, config: dict, use_api: bool):
        self.root = root
        self.config = config
        self.use_api = use_api

        # engines (created in the background)
        self.kb = KnowledgeBase(CONTEXT_DIR)
        self.index: Optional[SearchIndex] = None
        self.llm: Optional[GeminiClient] = None
        self.answer_engine: Optional[AnswerEngine] = None
        self.transcriber: Optional[AudioTranscriber] = None
        self.audio: Optional[AudioEngine] = None
        self.asr_model = None
        self.ready = False

        # UI state
        self.current_mode = "Interview"
        self.is_recording_meeting = False
        self.meeting_paragraphs: List[dict] = []
        self.meeting_current_speaker: Optional[str] = None
        self.meeting_buffer = ""
        self.meeting_seen: set = set()
        self.rendered_version = -1

        # answering state
        self.answer_generation = 0
        self.answer_lock = threading.Lock()
        self.last_answered_key = ""
        self.last_answered_time = 0.0
        self.answer_view: Dict[str, str] = {}
        self.auto_answer = bool(config.get("auto_answer", True))

        self._build_ui()
        self.ui(lambda: None)
        threading.Thread(target=self._bootstrap, name="bootstrap", daemon=True).start()
        self.root.after(200, self._ui_loop)

    # ------------------------------------------------------------------ utils
    def ui(self, fn, *args):
        """Run ``fn`` on the Tk thread (Tk is not thread-safe)."""
        try:
            self.root.after(0, lambda: fn(*args))
        except Exception:
            pass

    def set_status(self, text: str, color: str = "#8b949e"):
        self.ui(lambda: self.status_lbl.configure(text=text, text_color=color))

    def set_device_status(self, text: str, color: str = "#8b949e"):
        self.ui(lambda: self.device_lbl.configure(text=text, text_color=color))

    # -------------------------------------------------------------- bootstrap
    def _bootstrap(self):
        """Load knowledge, models and audio without blocking the window."""
        try:
            self.set_status("Loading knowledge base…", "#e8c84a")
            self._load_knowledge(initial=True)

            self.set_status(f"Loading speech model ({self.config.get('whisper_model', 'base.en')})…", "#e8c84a")
            import TranscriberModels

            self.asr_model = TranscriberModels.get_model(self.use_api, self.config)
            if self.config.get("asr_domain_prompt", True):
                try:
                    self.asr_model.set_vocabulary(self.kb.vocabulary_hint())
                except Exception:
                    pass
            self.transcriber = AudioTranscriber(self.asr_model, self.config, on_update=self._on_transcript_update,
                                                workers=int(self.config.get("asr_workers", 2)))
            self.transcriber.start()

            self.set_status("Opening audio devices…", "#e8c84a")
            self.audio = AudioEngine(self.config, on_segment=self._on_segment, on_status=self._on_audio_status)
            self.audio.start()
            self.ready = True
            self.set_status("Mode: Interview Copilot", "#7ee787")
            self.set_device_status(self.audio.status_text(), "#8b949e")
            self._write_answer(self._welcome_text())
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.set_status(f"Startup error: {exc}", "#f85149")
            self._write_answer(f"★ Startup error:\n{exc}\n\nCheck the console for details.")

    def _welcome_text(self) -> str:
        st = self.kb.stats.summary()
        hk = self.config.get("hotkey", "alt+q")
        ai = self.llm.describe() if self.llm else "AI: off"
        return (
            "★ Interview Mode Active.\n"
            f"Knowledge: {st}\n{ai} · ASR: {getattr(self.asr_model, 'name', '?')}"
            f" ({self.config.get('whisper_model', '')})\n"
            f"Auto-answer is {'ON' if self.auto_answer else 'OFF'} – questions from the interviewer are answered "
            f"automatically (Q&A file → HEAT → Gemini). Press [{hk.upper()}] or 'Answer' to force an answer."
        )

    def _load_knowledge(self, initial: bool = False):
        stats = self.kb.load()
        print(f"[Knowledge] {stats.summary()}")
        api_key = resolve_api_key(self.config)
        self.llm = GeminiClient(api_key, self.config.get("model_name", "gemini-2.5-flash"),
                                self.config.get("max_tokens", 300), self.config.get("temperature", 0.3),
                                self.config.get("embedding_model", "gemini-embedding-001"),
                                thinking=self.config.get("llm_thinking", "minimal"))
        if self.llm.last_error:
            print(f"[Gemini] {self.llm.last_error}")
        backend = GeminiEmbeddings(self.llm) if (self.llm.available and self.config.get("semantic_search", True)) else None
        self.index = SearchIndex(self.kb.qa_pairs, self.kb.passages, embedding_backend=backend, cache_dir=CONTEXT_DIR)
        if self.answer_engine is None:
            self.answer_engine = AnswerEngine(self.kb, self.index, self.llm, self.config)
        else:
            self.answer_engine.llm = self.llm
            self.answer_engine.rebuild(self.kb, self.index)
        if backend is not None and self.kb.qa_pairs:
            threading.Thread(target=self._build_semantic, name="semantic-index", daemon=True).start()
        if self.asr_model is not None and self.config.get("asr_domain_prompt", True):
            try:
                self.asr_model.set_vocabulary(self.kb.vocabulary_hint())
            except Exception:
                pass
        self.ui(lambda: self.kb_lbl.configure(text=f"📚 {stats.summary()}"))

    def _build_semantic(self):
        idx = self.index
        if idx is None:
            return
        t0 = time.perf_counter()
        ok = idx.build_semantic_index()
        print(f"[Semantic] index {'ready' if ok else 'unavailable'} ({time.perf_counter() - t0:.1f}s)")

    # ---------------------------------------------------------- audio events
    def _on_segment(self, segment: Segment):
        if self.transcriber is not None:
            self.transcriber.submit(segment)

    def _on_audio_status(self, level: str, message: str):
        print(f"[Audio] {message}")
        color = {"error": "#f85149", "warning": "#e8c84a"}.get(level, "#8b949e")
        if self.audio is not None:
            self.set_device_status(self.audio.status_text(), color)
        if level != "info" or "Reconnect" in message:
            self.set_status(message[:70], color)

    # ------------------------------------------------------ transcript events
    def _on_transcript_update(self, entry: TranscriptEntry):
        """Called from an ASR worker thread whenever a phrase (interim/final) is decoded."""
        if entry.who != "Speaker" or self.current_mode != "Interview" or not self.auto_answer:
            return
        if not entry.is_final and question_score(entry.text) < 0.75:
            return
        self._maybe_auto_answer()

    def _maybe_auto_answer(self):
        if self.transcriber is None or self.answer_engine is None:
            return
        window = float(self.config.get("answer_window_seconds", 20))
        phrases = self.transcriber.recent_phrases("Speaker", seconds=window)
        detected = extract_question(phrases, window_seconds=window,
                                    min_score=float(self.config.get("auto_answer_min_score", 0.45)))
        if not detected:
            return
        key = normalize_for_dedupe(detected.text)
        with self.answer_lock:
            if key == self.last_answered_key:
                return
            # The same question growing by a few words (interim → final) is not a new question
            if self.last_answered_key and (key.startswith(self.last_answered_key) or self.last_answered_key.startswith(key)):
                extra = abs(len(key.split()) - len(self.last_answered_key.split()))
                if extra < 3 and time.time() - self.last_answered_time < 8:
                    return
            self.last_answered_key = key
            self.last_answered_time = time.time()
        self._run_answer(detected.text, detected.context, auto=True)

    # ------------------------------------------------------------- answering
    def _run_answer(self, question: str, context: str = "", auto: bool = False, force_ai: bool = False):
        if self.answer_engine is None:
            self._write_answer("★ Still loading – one moment…")
            return
        with self.answer_lock:
            self.answer_generation += 1
            gen = self.answer_generation
        self.answer_view = {"question": question, "auto": "auto" if auto else "manual"}

        def is_cancelled():
            return gen != self.answer_generation

        def on_event(ev: AnswerEvent):
            if ev.generation != self.answer_generation:
                return
            self.ui(self._render_answer_event, ev)

        threading.Thread(target=self.answer_engine.answer,
                         args=(question, context, on_event, gen, is_cancelled, force_ai),
                         name="answer", daemon=True).start()

    def trigger_answer(self, force_ai: bool = False):
        """Hotkey / button: answer the latest interviewer question (or last phrase)."""
        if self.current_mode != "Interview":
            return
        if self.transcriber is None or self.answer_engine is None:
            self._write_answer("★ Still loading – one moment…")
            return
        window = float(self.config.get("answer_window_seconds", 20)) * 2
        phrases = self.transcriber.recent_phrases("Speaker", seconds=window)
        detected = extract_question(phrases, window_seconds=window, min_score=0.3)
        if detected:
            question, context = detected.text, detected.context
        elif phrases:
            question, context = " ".join(p[1] for p in phrases[-2:]), ""
        else:
            own = self.transcriber.recent_phrases("You", seconds=window)
            if own:
                question, context = own[-1][1], ""
            else:
                self._write_answer("★ Nothing heard from the interviewer yet.")
                return
        with self.answer_lock:
            self.last_answered_key = normalize_for_dedupe(question)
            self.last_answered_time = time.time()
        self._run_answer(question, context, auto=False, force_ai=force_ai)

    # --------------------------------------------------------- answer render
    def _write_answer(self, text: str):
        self.ui(self._set_textbox, self.answer_box, text)

    def _set_textbox(self, box, text):
        box.delete("0.0", "end")
        box.insert("0.0", text)

    def _render_answer_event(self, ev: AnswerEvent):
        v = self.answer_view
        if ev.kind == "start":
            v.update(header=f"★ Q: “{ev.text}”", body="⏳ searching your Q&A / HEAT files…", ai="", source="")
        elif ev.kind == "local":
            v["source"] = f"📚 {ev.source} · match {ev.confidence:.0%} · {ev.elapsed_ms:.0f} ms"
            v["body"] = f"≈ {ev.matched_question}\n\n{ev.text}"
            v["ai"] = ""
        elif ev.kind == "candidate":
            v["source"] = f"🔎 closest {ev.source} match {ev.confidence:.0%} · {ev.elapsed_ms:.0f} ms"
            v["body"] = f"≈ {ev.matched_question}\n\n{ev.text}"
        elif ev.kind == "ai_start":
            if not v.get("body") or v.get("body", "").startswith("⏳"):
                v["body"] = ""
            v["ai"] = "✨ Gemini:\n"
        elif ev.kind == "ai_chunk":
            v["ai"] = v.get("ai", "✨ Gemini:\n") + ev.text
        elif ev.kind == "ai_done":
            if ev.elapsed_ms:
                v["source"] = (v.get("source") or "") + (f"  ·  ✨ AI done in {ev.elapsed_ms / 1000:.1f}s")
        elif ev.kind == "info":
            v["body"] = ev.text
        elif ev.kind == "error":
            v["ai"] = (v.get("ai") or "") + f"\n[{ev.text}]"
        parts = [v.get("header", "★ Abujami Assistant")]
        if v.get("source"):
            parts.append(v["source"])
        if v.get("body"):
            parts.append("\n" + v["body"])
        if v.get("ai"):
            parts.append("\n" + v["ai"])
        text = "\n".join(parts)
        self.answer_box.delete("0.0", "end")
        self.answer_box.insert("0.0", text)

    # ---------------------------------------------------------------- UI loop
    def _ui_loop(self):
        try:
            self._ui_tick()
        except Exception as exc:  # pragma: no cover - keep the loop alive
            print(f"[UI] {exc}")
        self.root.after(200, self._ui_loop)

    def _ui_tick(self):
        if self.transcriber is None:
            return
        if self.config.get("show_visualizer", True) and self.audio is not None:
            lv = self.audio.levels()
            self.mic_bar.set(lv.get("You", 0.0))
            self.spk_bar.set(lv.get("Speaker", 0.0))
        if self.transcriber.version == self.rendered_version:
            return
        self.rendered_version = self.transcriber.version

        if self.current_mode == "Interview":
            self._set_textbox(self.transcript_box, self.transcriber.get_transcript(newest_first=True))
        elif self.current_mode == "Meeting" and self.is_recording_meeting:
            for e in self.transcriber.get_entries(final_only=True):
                if e.phrase_id in self.meeting_seen or len(e.text) < 4:
                    continue
                self.meeting_seen.add(e.phrase_id)
                spk = "Candidate (You)" if e.who == "You" else "Remote (Teams)"
                now_str = datetime.fromtimestamp(e.end_time).strftime("%H:%M:%S")
                if self.meeting_current_speaker == spk:
                    self.meeting_buffer += f" {e.text}"
                    self.transcript_box.insert("end", f" {e.text}")
                else:
                    self._flush_meeting_paragraph()
                    self.meeting_current_speaker = spk
                    self.meeting_buffer = e.text
                    label = "[Remote / Teams]" if "Teams" in spk else "[Candidate / You]"
                    self.transcript_box.insert("end", f"\n\n[{now_str}] {label}:\n{e.text}")
                    self.meeting_paragraphs.append({"speaker": label, "time": now_str, "text": e.text})
                self.transcript_box.see("end")
            self.status_lbl.configure(text=f"● Recording ({len(self.meeting_paragraphs)} paragraphs)", text_color="#f85149")

    def _flush_meeting_paragraph(self):
        if self.meeting_paragraphs and self.meeting_buffer:
            self.meeting_paragraphs[-1]["text"] = self.meeting_buffer.strip()

    # ------------------------------------------------------------ mode logic
    def toggle_mode(self, selected_mode: str):
        self.current_mode = selected_mode
        hk = self.config.get("hotkey", "alt+q")
        if selected_mode == "Interview":
            self.is_recording_meeting = False
            self.btn_action.configure(text=f"Answer ({hk})", fg_color="#1f6feb", hover_color="#388bfd")
            self.status_lbl.configure(text="Mode: Interview Copilot", text_color="#7ee787")
            self.answer_box.grid()
            self._write_answer(self._welcome_text() if self.ready else "★ Loading…")
            self.rendered_version = -1
        else:
            self.btn_action.configure(text="⏺ Start Meeting", fg_color="#238636", hover_color="#2ea043")
            self.status_lbl.configure(text="Mode: Meeting Logger (Paragraphs)", text_color="#58a6ff")
            self.answer_box.grid_remove()
            self._set_textbox(self.transcript_box, "")

    def handle_main_action(self):
        if self.current_mode == "Interview":
            self.trigger_answer()
            return
        if not self.is_recording_meeting:
            self.is_recording_meeting = True
            self.meeting_paragraphs.clear()
            self.meeting_seen.clear()
            self.meeting_current_speaker = None
            self.meeting_buffer = ""
            if self.transcriber:
                self.transcriber.clear_transcript_data()
            self._set_textbox(self.transcript_box, "")
            self.btn_action.configure(text="⏹ Stop & Save Word", fg_color="#da3633", hover_color="#b62324")
            self.status_lbl.configure(text="● Recording Meeting...", text_color="#f85149")
        else:
            self.is_recording_meeting = False
            self._flush_meeting_paragraph()
            self.btn_action.configure(text="⏺ Start Meeting", fg_color="#238636", hover_color="#2ea043")
            self.status_lbl.configure(text="Saving Word...", text_color="#e8c84a")
            paragraphs = [dict(p) for p in self.meeting_paragraphs]

            def do_save():
                path = export_meeting_to_word(paragraphs)
                if path:
                    self.set_status("Word Saved & Opened!", "#7ee787")
                    open_path(path)
                else:
                    self.set_status("No speech recorded.", "#8b949e")

            threading.Thread(target=do_save, daemon=True).start()

    def clear_context(self):
        self.meeting_current_speaker = None
        self.meeting_buffer = ""
        self.meeting_paragraphs.clear()
        self.meeting_seen.clear()
        if self.transcriber:
            self.transcriber.clear_transcript_data()
        with self.answer_lock:
            self.answer_generation += 1
            self.last_answered_key = ""
        if self.answer_engine:
            self.answer_engine.history.clear()
        self._set_textbox(self.transcript_box, "")
        self._write_answer("★ Abujami Assistant: Ready.")
        self.status_lbl.configure(text="Cleared", text_color="#7ee787")

    def toggle_auto_answer(self):
        self.auto_answer = bool(self.auto_switch.get())
        self.config["auto_answer"] = self.auto_answer
        save_config(self.config)

    def reconnect_audio(self):
        if self.audio is None:
            return
        self.set_status("🔄 Reconnecting audio…", "#e8c84a")
        threading.Thread(target=self.audio.reconnect, args=("manual",), daemon=True).start()

    # ----------------------------------------------------------------- settings
    def open_settings_window(self):
        win = ctk.CTkToplevel(self.root)
        win.title("Abujami Assistant Settings")
        win.geometry("600x720")
        win.attributes("-topmost", True)
        win.configure(fg_color="#161b22")
        win.grab_set()

        ctk.CTkLabel(win, text="⚙ Settings & Knowledge Base", font=("Segoe UI", 16, "bold"), text_color="#58a6ff").pack(pady=10)
        container = ctk.CTkScrollableFrame(win, fg_color="#0d1117", width=560, height=580)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        def label(text, color="#c9d1d9", size=12, bold=True):
            ctk.CTkLabel(container, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                         text_color=color).pack(anchor="w", padx=10, pady=(8, 2))

        label("Gemini API Key:")
        entry_api = ctk.CTkEntry(container, width=520, show="*")
        entry_api.pack(padx=10, pady=(0, 6))
        entry_api.insert(0, self.config.get("gemini_api_key", ""))

        label("Gemini model:")
        entry_model = ctk.CTkEntry(container, width=520)
        entry_model.pack(padx=10, pady=(0, 6))
        entry_model.insert(0, self.config.get("model_name", "gemini-2.5-flash"))

        row = ctk.CTkFrame(container, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row, text="Speech model:", text_color="#c9d1d9", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        whisper_var = ctk.StringVar(value=str(self.config.get("whisper_model", "base.en")))
        ctk.CTkOptionMenu(row, values=WHISPER_CHOICES, variable=whisper_var, width=170).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(row, text="Audio device role:", text_color="#c9d1d9", font=("Segoe UI", 12, "bold")).grid(row=0, column=2, sticky="w", padx=(16, 8))
        role_var = ctk.StringVar(value=str(self.config.get("device_role", "auto")))
        ctk.CTkOptionMenu(row, values=DEVICE_ROLE_CHOICES, variable=role_var, width=150).grid(row=0, column=3, sticky="w")

        switches = ctk.CTkFrame(container, fg_color="transparent")
        switches.pack(fill="x", padx=10, pady=4)
        auto_var = ctk.BooleanVar(value=bool(self.config.get("auto_answer", True)))
        ctk.CTkSwitch(switches, text="Auto-answer detected questions", variable=auto_var).grid(row=0, column=0, sticky="w", padx=4, pady=3)
        sem_var = ctk.BooleanVar(value=bool(self.config.get("semantic_search", True)))
        ctk.CTkSwitch(switches, text="Semantic search (Gemini embeddings)", variable=sem_var).grid(row=1, column=0, sticky="w", padx=4, pady=3)
        refine_var = ctk.BooleanVar(value=bool(self.config.get("ai_refine_local_answers", False)))
        ctk.CTkSwitch(switches, text="Let Gemini rephrase answers found in the Q&A file", variable=refine_var).grid(row=2, column=0, sticky="w", padx=4, pady=3)
        mic_var = ctk.BooleanVar(value=bool(self.config.get("transcribe_mic", True)))
        ctk.CTkSwitch(switches, text="Transcribe my microphone too (needed for Meeting mode)", variable=mic_var).grid(row=3, column=0, sticky="w", padx=4, pady=3)

        label("📚 Knowledge files (context folder):", "#58a6ff", 13)
        stats_lbl = ctk.CTkLabel(container, text=self.kb.stats.summary(), text_color="#8b949e", font=("Segoe UI", 11), wraplength=520, justify="left")
        stats_lbl.pack(anchor="w", padx=10)
        box = ctk.CTkFrame(container, fg_color="#161b22")
        box.pack(fill="x", padx=10, pady=4)

        def upload_file(prefix, lbl):
            path = filedialog.askopenfilename(filetypes=[("Documents", "*.docx *.pdf *.txt *.md")])
            if not path:
                return
            ext = os.path.splitext(path)[1]
            os.makedirs(CONTEXT_DIR, exist_ok=True)
            for old in glob.glob(os.path.join(CONTEXT_DIR, f"{prefix}.*")):
                try:
                    os.remove(old)
                except Exception:
                    pass
            shutil.copy(path, os.path.join(CONTEXT_DIR, f"{prefix}{ext}"))
            lbl.configure(text=f"Loaded: {os.path.basename(path)}", text_color="#7ee787")

        buttons = [
            ("❓ Upload Q&A file", "qa_bank", "#1f6feb", "Q&A bank – answered first"),
            ("🛡️ Upload HEAT", "heat_security", "#8957e5", "HEAT training – searched second"),
            ("📄 Upload CV", "cv_candidate", "#21262d", "CV Status"),
            ("🎯 Upload JD", "jd_target", "#21262d", "JD Status"),
        ]
        for r, (text, prefix, color, status) in enumerate(buttons):
            lbl = ctk.CTkLabel(box, text=status, text_color="#8b949e", font=("Segoe UI", 11))
            ctk.CTkButton(box, text=text, width=160, fg_color=color,
                          command=lambda p=prefix, l=lbl: upload_file(p, l)).grid(row=r, column=0, padx=6, pady=5, sticky="w")
            lbl.grid(row=r, column=1, padx=6, pady=5, sticky="w")

        tools = ctk.CTkFrame(container, fg_color="transparent")
        tools.pack(fill="x", padx=10, pady=(12, 6))
        ctk.CTkButton(tools, text="📁 Open context folder", width=170, fg_color="#21262d",
                      command=lambda: (os.makedirs(CONTEXT_DIR, exist_ok=True), open_path(CONTEXT_DIR))).grid(row=0, column=0, padx=4)
        ctk.CTkButton(tools, text="📁 Recordings", width=120, fg_color="#21262d",
                      command=lambda: (os.makedirs(RECORDINGS_DIR, exist_ok=True), open_path(RECORDINGS_DIR))).grid(row=0, column=1, padx=4)
        ctk.CTkButton(tools, text="🔄 Reconnect audio", width=150, fg_color="#21262d",
                      command=self.reconnect_audio).grid(row=0, column=2, padx=4)

        def save_settings():
            old_model = self.config.get("whisper_model")
            old_mic = self.config.get("transcribe_mic", True)
            old_role = self.config.get("device_role", "auto")
            self.config["gemini_api_key"] = entry_api.get().strip()
            self.config["model_name"] = entry_model.get().strip() or "gemini-2.5-flash"
            self.config["whisper_model"] = whisper_var.get()
            self.config["device_role"] = role_var.get()
            self.config["auto_answer"] = bool(auto_var.get())
            self.config["semantic_search"] = bool(sem_var.get())
            self.config["ai_refine_local_answers"] = bool(refine_var.get())
            self.config["transcribe_mic"] = bool(mic_var.get())
            self.auto_answer = self.config["auto_answer"]
            if self.auto_answer:
                self.auto_switch.select()
            else:
                self.auto_switch.deselect()
            save_config(self.config)
            win.destroy()
            self.set_status("Syncing knowledge…", "#e8c84a")

            def sync():
                self._load_knowledge()
                if self.config.get("whisper_model") != old_model and self.transcriber is not None and not self.use_api:
                    try:
                        import TranscriberModels

                        self.set_status(f"Loading speech model ({self.config['whisper_model']})…", "#e8c84a")
                        new_model = TranscriberModels.get_model(False, self.config)
                        new_model.set_vocabulary(self.kb.vocabulary_hint())
                        self.transcriber.audio_model = new_model
                        self.asr_model = new_model
                    except Exception as exc:
                        self.set_status(f"Speech model error: {exc}", "#f85149")
                if self.audio is not None and (old_mic != self.config.get("transcribe_mic") or old_role != self.config.get("device_role")):
                    self.audio.enabled_sources["You"] = bool(self.config.get("transcribe_mic", True))
                    self.audio.reconnect("settings changed")
                self.set_status("Settings saved & knowledge synced", "#7ee787")
                self._write_answer(self._welcome_text())

            threading.Thread(target=sync, daemon=True).start()

        ctk.CTkButton(win, text="Save & Sync", fg_color="#238636", command=save_settings).pack(pady=10)

    # -------------------------------------------------------------------- UI
    def _build_ui(self):
        root = self.root
        ctk.set_appearance_mode("dark")
        root.title("Abujami Assistant (Dual Engine)")
        root.geometry("980x540+180+40")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", float(self.config.get("alpha", 0.82)))
        except Exception:
            pass
        root.configure(fg_color="#0d1117")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=0)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)

        top_bar = ctk.CTkFrame(root, height=38, fg_color="#161b22")
        top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))

        self.mode_selector = ctk.CTkSegmentedButton(top_bar, values=["Interview", "Meeting"], command=self.toggle_mode)
        self.mode_selector.set("Interview")
        self.mode_selector.pack(side="left", padx=8, pady=5)

        hk = self.config.get("hotkey", "alt+q")
        self.btn_action = ctk.CTkButton(top_bar, text=f"Answer ({hk})", width=120, height=26,
                                        font=("Segoe UI", 11, "bold"), fg_color="#1f6feb", command=self.handle_main_action)
        self.btn_action.pack(side="left", padx=5)

        ctk.CTkButton(top_bar, text="✨ AI", width=50, height=26, fg_color="#21262d", hover_color="#8957e5",
                      command=lambda: self.trigger_answer(force_ai=True)).pack(side="left", padx=3)
        ctk.CTkButton(top_bar, text="Clear", width=55, height=26, fg_color="#21262d", hover_color="#da3633",
                      command=self.clear_context).pack(side="left", padx=5)
        ctk.CTkButton(top_bar, text="⚙ Settings", width=75, height=26, fg_color="#21262d",
                      command=self.open_settings_window).pack(side="left", padx=5)

        self.auto_switch = ctk.CTkSwitch(top_bar, text="Auto", width=60, command=self.toggle_auto_answer,
                                         font=("Segoe UI", 11, "bold"))
        if self.auto_answer:
            self.auto_switch.select()
        self.auto_switch.pack(side="left", padx=6)

        self.mic_bar = ctk.CTkProgressBar(top_bar, width=40, height=8, progress_color="#39ff14")
        self.mic_bar.set(0)
        self.mic_bar.pack(side="left", padx=6)
        self.spk_bar = ctk.CTkProgressBar(top_bar, width=40, height=8, progress_color="#58a6ff")
        self.spk_bar.set(0)
        self.spk_bar.pack(side="left", padx=6)

        self.stealth_badge = ctk.CTkLabel(top_bar, text="🛡️ Stealth: ON", font=("Segoe UI", 11, "bold"), text_color="#39ff14")
        self.stealth_badge.pack(side="left", padx=10)

        self.status_lbl = ctk.CTkLabel(top_bar, text="Starting…", font=("Segoe UI", 11), text_color="#e8c84a")
        self.status_lbl.pack(side="right", padx=10)

        info_bar = ctk.CTkFrame(root, height=22, fg_color="transparent")
        info_bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 2))
        self.device_lbl = ctk.CTkLabel(info_bar, text="🎤 —   |   🔊 —", font=("Segoe UI", 10), text_color="#8b949e")
        self.device_lbl.pack(side="left")
        ctk.CTkButton(info_bar, text="🔄", width=28, height=20, fg_color="#21262d", command=self.reconnect_audio).pack(side="left", padx=6)
        self.kb_lbl = ctk.CTkLabel(info_bar, text="📚 loading knowledge…", font=("Segoe UI", 10), text_color="#8b949e")
        self.kb_lbl.pack(side="right")

        f1 = ctk.CTkFrame(root, fg_color="#161b22")
        f1.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        self.transcript_box = ctk.CTkTextbox(f1, font=("Consolas", 12), text_color="#c9d1d9", fg_color="transparent", wrap="word")
        self.transcript_box.pack(fill="both", expand=True, padx=6, pady=6)

        f2 = ctk.CTkFrame(root, fg_color="#161b22")
        f2.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        self.answer_box = ctk.CTkTextbox(f2, font=("Segoe UI", 13), text_color="#f0f6fc", fg_color="transparent", wrap="word")
        self.answer_box.pack(fill="both", expand=True, padx=8, pady=8)
        self._set_textbox(self.answer_box, "★ Loading knowledge base, speech model and audio devices…")

        root.bind("<Return>", lambda e: self.handle_main_action())
        if keyboard is not None:
            try:
                keyboard.add_hotkey(hk, lambda: self.ui(self.handle_main_action))
            except Exception as exc:
                print(f"[Hotkey] {exc}")

        stealth_on = False
        if self.config.get("stealth_mode", True):
            try:
                hwnd = ctypes.windll.user32.GetParent(root.winfo_id())  # type: ignore[attr-defined]
                stealth_on = set_stealth_mode(hwnd, True)
            except Exception:
                stealth_on = False
        self.stealth_badge.configure(text="🛡️ Stealth: ON" if stealth_on else "🛡️ Stealth: OFF",
                                     text_color="#39ff14" if stealth_on else "#8b949e")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            if self.audio:
                self.audio.stop()
            if self.transcriber:
                self.transcriber.stop()
        finally:
            self.root.destroy()


def main():
    config = load_config()
    use_api = "--api" in sys.argv
    root = ctk.CTk()
    AbujamiApp(root, config, use_api)
    root.mainloop()


if __name__ == "__main__":
    main()
