import os
import glob
import json
import threading
import queue
import time
import sys
import subprocess
import ctypes
import keyboard
import numpy as np
import shutil
from datetime import datetime

from AudioTranscriber import AudioTranscriber
import customtkinter as ctk
import AudioRecorder 
import TranscriberModels
import google.generativeai as genai
from tkinter import filedialog, messagebox

try:
    import docx
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

CONFIG_FILE = "config.json"
RECORDINGS_DIR = "recordings"
CONTEXT_DIR = "context"

DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "model_name": "gemini-3.6-flash",
    "alpha": 0.82,
    "stealth_mode": True,
    "hotkey": "alt+q",
    "max_tokens": 450,
    "temperature": 0.25
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

app_config = load_config()

knowledge_base = {"CV": "", "JD": "", "QA": "", "HEAT": "", "EXTRA": ""}
gemini_chat_session = None
chat_lock = threading.Lock()

# إدارة الأوضاع وتسجيل الفقرات
current_mode = "Interview"  # "Interview" أو "Meeting"
is_recording_meeting = False
meeting_paragraphs = []
current_speaker = None
current_text_buffer = ""
seen_lines = set()

WDA_EXCLUDEFROMCAPTURE = 0x00000011

def set_stealth_mode(hwnd, enable=True):
    try:
        affinity = WDA_EXCLUDEFROMCAPTURE if enable else 0
        return ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity) != 0
    except Exception:
        return False

def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".docx" and DOCX_AVAILABLE:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        except Exception:
            pass
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            pass
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            pass
    return text.strip()

def load_all_categorized_context():
    global knowledge_base
    os.makedirs(CONTEXT_DIR, exist_ok=True)
    categories = {
        "CV": ["cv_*", "cv.*", "resume*"],
        "JD": ["jd_*", "jd.*", "job*"],
        "QA": ["qa_*", "qa.*", "interview*"],
        "HEAT": ["heat_*", "heat.*", "security_training*"],
        "EXTRA": ["extra_*", "ref_*", "doc_*"]
    }
    for cat, patterns in categories.items():
        cat_text = []
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(CONTEXT_DIR, pattern)):
                content = extract_text_from_file(filepath)
                if content:
                    cat_text.append(f"[{cat} DOC: {os.path.basename(filepath)}]\n{content}")
        knowledge_base[cat] = "\n\n".join(cat_text)
    init_live_gemini_session()

def init_live_gemini_session():
    global gemini_chat_session
    if not app_config["gemini_api_key"]:
        return
    try:
        genai.configure(api_key=app_config["gemini_api_key"])
        full_system_context = (
            "You are Abujami Assistant — elite real-time copilot.\n"
            f"--- CV ---\n{knowledge_base['CV'] or 'None'}\n\n"
            f"--- JD ---\n{knowledge_base['JD'] or 'None'}\n\n"
            f"--- HEAT ---\n{knowledge_base['HEAT'] or 'None'}\n\n"
            f"--- EXTRA ---\n{knowledge_base['EXTRA'] or 'None'}\n\n"
            "RULES: Speak as candidate in first person ('I'). Answer in 3-4 punchy bullet points directly."
        )
        generation_config = {
            "max_output_tokens": int(app_config.get("max_tokens", 450)),
            "temperature": float(app_config.get("temperature", 0.25))
        }
        model = genai.GenerativeModel(
            app_config.get("model_name", "gemini-3.6-flash"),
            system_instruction=full_system_context,
            generation_config=generation_config
        )
        with chat_lock:
            gemini_chat_session = model.start_chat(history=[])
    except Exception as e:
        print(f"[Gemini Init Error] {e}")

load_all_categorized_context()

def write_in_textbox(textbox, text):
    textbox.delete("0.0", "end")
    textbox.insert("0.0", text)

def append_in_textbox(textbox, text):
    textbox.insert("end", text)
    textbox.see("end")

def flush_current_meeting_paragraph():
    global current_text_buffer, meeting_paragraphs
    if meeting_paragraphs and current_text_buffer:
        meeting_paragraphs[-1]["text"] = current_text_buffer.strip()

def export_meeting_to_word():
    flush_current_meeting_paragraph()
    if not meeting_paragraphs:
        return None

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(RECORDINGS_DIR, f"Meeting_{ts}.docx")

    if DOCX_AVAILABLE:
        doc = docx.Document()
        title = doc.add_heading("Meeting Record & Transcript", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
        doc.add_paragraph("─" * 45)

        for p in meeting_paragraphs:
            p_head = doc.add_paragraph()
            r_time = p_head.add_run(f"[{p['time']}] ")
            r_time.font.color.rgb = RGBColor(120, 120, 120)
            r_spk = p_head.add_run(f"{p['speaker']}:")
            r_spk.bold = True
            r_spk.font.color.rgb = RGBColor(31, 110, 235) if "Teams" in p["speaker"] else RGBColor(35, 134, 54)

            p_body = doc.add_paragraph(p["text"].strip())
            p_body.paragraph_format.space_after = Pt(10)

        doc.save(filepath)
    else:
        filepath = os.path.join(RECORDINGS_DIR, f"Meeting_{ts}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            for p in meeting_paragraphs:
                f.write(f"[{p['time']}] {p['speaker']}:\n{p['text']}\n\n")

    return filepath

audio_levels = {"mic": 0.0, "speaker": 0.0}

def update_ui_loop(transcriber, transcript_box, answer_box, mic_bar, spk_bar, status_lbl):
    global current_speaker, current_text_buffer

    raw_transcript = transcriber.get_transcript()

    if current_mode == "Interview":
        # وضع المقابلة: إظهار النص الحرفي السريع مباشرة
        write_in_textbox(transcript_box, raw_transcript)

    elif current_mode == "Meeting" and is_recording_meeting:
        # وضع الاجتماع: تجميع الكلام المستمر في فقرات عادية بدون تقطيع
        lines = [l.strip() for l in raw_transcript.split("\n") if l.strip()]
        for line in lines:
            if len(line) < 4:
                continue

            if line.startswith("You:"):
                spk = "Candidate (You)"
                text = line[4:].strip()
            elif line.startswith("Speaker:"):
                spk = "Remote (Teams)"
                text = line[8:].strip()
            else:
                spk = "Remote (Teams)"
                text = line

            # بصمة الجملة لمنع التكرار
            sig = f"{spk}::{text.lower()}"
            if sig not in seen_lines:
                seen_lines.add(sig)
                now_str = datetime.now().strftime("%H:%M:%S")

                if current_speaker == spk:
                    # نفس المتحدث -> دمج النص كفقرة عادية متصلة
                    current_text_buffer += f" {text}"
                    transcript_box.insert("end", f" {text}")
                else:
                    # متحدث جديد -> إنزال سطر، طباعة الوقت والمتحدث، ثم نص الفقرة
                    flush_current_meeting_paragraph()
                    current_speaker = spk
                    current_text_buffer = text
                    label = "[Remote / Teams]" if "Teams" in spk else "[Candidate / You]"
                    header = f"\n\n[{now_str}] {label}:\n"
                    transcript_box.insert("end", header + text)
                    meeting_paragraphs.append({"speaker": label, "time": now_str, "text": text})

                transcript_box.see("end")
                status_lbl.configure(text=f"● Recording ({len(meeting_paragraphs)} paragraphs)", text_color="#f85149")

    if app_config.get("show_visualizer", True):
        mic_bar.set(audio_levels["mic"])
        spk_bar.set(audio_levels["speaker"])

    transcript_box.after(250, update_ui_loop, transcriber, transcript_box, answer_box, mic_bar, spk_bar, status_lbl)

def toggle_mode(selected_mode, btn_action, transcript_box, answer_box, status_lbl, transcriber):
    global current_mode, is_recording_meeting
    current_mode = selected_mode

    if current_mode == "Interview":
        is_recording_meeting = False
        btn_action.configure(text=f"Answer ({app_config.get('hotkey', 'alt+q')})", fg_color="#1f6feb", hover_color="#388bfd")
        status_lbl.configure(text="Mode: Interview Copilot", text_color="#7ee787")
        answer_box.grid()
        write_in_textbox(answer_box, "★ Interview Mode Active: Press [Alt + Q] or click 'Answer' for instant talking points.")
    else:
        btn_action.configure(text="⏺ Start Meeting", fg_color="#238636", hover_color="#2ea043")
        status_lbl.configure(text="Mode: Meeting Logger (Paragraphs)", text_color="#58a6ff")
        answer_box.grid_remove()  # إخفاء صندوق الإجابات لتوسيع شاشة الاجتماع بالكامل
        write_in_textbox(transcript_box, "")

def handle_main_action(transcriber, btn_action, status_lbl, transcript_box, answer_box):
    global is_recording_meeting, current_speaker, current_text_buffer
    if current_mode == "Interview":
        trigger_answer_threaded(transcriber, answer_box)
    else:
        # وضع الاجتماع: بدء أو إيقاف وحفظ Word
        if not is_recording_meeting:
            is_recording_meeting = True
            meeting_paragraphs.clear()
            seen_lines.clear()
            current_speaker = None
            current_text_buffer = ""
            transcriber.clear_transcript_data()
            write_in_textbox(transcript_box, "")
            btn_action.configure(text="⏹ Stop & Save Word", fg_color="#da3633", hover_color="#b62324")
            status_lbl.configure(text="● Recording Meeting...", text_color="#f85149")
        else:
            is_recording_meeting = False
            btn_action.configure(text="⏺ Start Meeting", fg_color="#238636", hover_color="#2ea043")
            status_lbl.configure(text="Saving Word...", text_color="#e8c84a")
            def do_save():
                path = export_meeting_to_word()
                if path:
                    status_lbl.configure(text="Word Saved & Opened!", text_color="#7ee787")
                    try: os.startfile(os.path.abspath(path))
                    except: pass
                else:
                    status_lbl.configure(text="No speech recorded.", text_color="#8b949e")
            threading.Thread(target=do_save, daemon=True).start()

def stream_gemini_suggestion(transcript_text, answer_box):
    global gemini_chat_session
    if not gemini_chat_session:
        init_live_gemini_session()
        if not gemini_chat_session:
            write_in_textbox(answer_box, "★ Error: Configure API Key in Settings ⚙")
            return

    trimmed_cue = transcript_text[-500:].strip() if len(transcript_text) > 500 else transcript_text.strip()
    if not trimmed_cue:
        return

    write_in_textbox(answer_box, "★ Abujami Assistant:\n")
    prompt = f"Live Interview Cue:\n\"{trimmed_cue}\"\n\nGenerate the candidate's exact 3-4 bullet response immediately:"

    try:
        with chat_lock:
            stream = gemini_chat_session.send_message(prompt, stream=True)
            for chunk in stream:
                if chunk.text:
                    append_in_textbox(answer_box, chunk.text)
    except Exception as e:
        append_in_textbox(answer_box, f"\n[Error: {e}]")

def trigger_answer_threaded(transcriber, answer_box):
    if current_mode != "Interview":
        return
    text = transcriber.get_transcript()
    threading.Thread(target=stream_gemini_suggestion, args=(text, answer_box), daemon=True).start()

def clear_context(transcriber, speaker_queue, mic_queue, transcript_box, answer_box, status_lbl):
    global current_speaker, current_text_buffer
    current_speaker = None
    current_text_buffer = ""
    meeting_paragraphs.clear()
    seen_lines.clear()
    transcriber.clear_transcript_data()
    write_in_textbox(transcript_box, "")
    write_in_textbox(answer_box, "★ Abujami Assistant: Ready.")
    with speaker_queue.mutex: speaker_queue.queue.clear()
    with mic_queue.mutex: mic_queue.queue.clear()
    init_live_gemini_session()
    status_lbl.configure(text="Cleared", text_color="#7ee787")

def open_settings_window(root, transcriber, answer_box, stealth_badge):
    win = ctk.CTkToplevel(root)
    win.title("Abujami Assistant Settings")
    win.geometry("560x650")
    win.attributes("-topmost", True)
    win.configure(fg_color="#161b22")
    win.grab_set()

    ctk.CTkLabel(win, text="⚙ Settings & Knowledge Base", font=("Segoe UI", 16, "bold"), text_color="#58a6ff").pack(pady=10)
    container = ctk.CTkScrollableFrame(win, fg_color="#0d1117", width=520, height=520)
    container.pack(fill="both", expand=True, padx=10, pady=5)

    ctk.CTkLabel(container, text="Gemini API Key:", font=("Segoe UI", 12, "bold"), text_color="#c9d1d9").pack(anchor="w", padx=10, pady=(8, 2))
    entry_api = ctk.CTkEntry(container, width=480, show="*")
    entry_api.pack(padx=10, pady=(0, 10))
    entry_api.insert(0, app_config.get("gemini_api_key", ""))

    ctk.CTkLabel(container, text="📚 Upload Knowledge Files:", font=("Segoe UI", 13, "bold"), text_color="#58a6ff").pack(anchor="w", padx=10, pady=(10, 4))
    box = ctk.CTkFrame(container, fg_color="#161b22")
    box.pack(fill="x", padx=10, pady=4)

    def upload_file(prefix, lbl):
        path = filedialog.askopenfilename(filetypes=[("Documents", "*.docx *.pdf *.txt")])
        if path:
            ext = os.path.splitext(path)[1]
            for old in glob.glob(os.path.join(CONTEXT_DIR, f"{prefix}.*")):
                try: os.remove(old)
                except: pass
            shutil.copy(path, os.path.join(CONTEXT_DIR, f"{prefix}{ext}"))
            lbl.configure(text=f"Loaded: {os.path.basename(path)}", text_color="#7ee787")

    # أزرار الرفع الأربعة
    btn_cv = ctk.CTkButton(box, text="📄 Upload CV", width=140, fg_color="#21262d", command=lambda: upload_file("cv_candidate", lbl_cv))
    btn_cv.grid(row=0, column=0, padx=6, pady=6)
    lbl_cv = ctk.CTkLabel(box, text="CV Status", text_color="#8b949e", font=("Segoe UI", 11))
    lbl_cv.grid(row=0, column=1, padx=6, pady=6, sticky="w")

    btn_jd = ctk.CTkButton(box, text="🎯 Upload JD", width=140, fg_color="#21262d", command=lambda: upload_file("jd_target", lbl_jd))
    btn_jd.grid(row=1, column=0, padx=6, pady=6)
    lbl_jd = ctk.CTkLabel(box, text="JD Status", text_color="#8b949e", font=("Segoe UI", 11))
    lbl_jd.grid(row=1, column=1, padx=6, pady=6, sticky="w")

    btn_heat = ctk.CTkButton(box, text="🛡️ Upload HEAT", width=140, fg_color="#8957e5", command=lambda: upload_file("heat_security", lbl_heat))
    btn_heat.grid(row=2, column=0, padx=6, pady=6)
    lbl_heat = ctk.CTkLabel(box, text="HEAT Status", text_color="#8b949e", font=("Segoe UI", 11))
    lbl_heat.grid(row=2, column=1, padx=6, pady=6, sticky="w")

    btn_open_rec = ctk.CTkButton(container, text="📁 Open Word Recordings Folder", width=220, fg_color="#21262d", command=lambda: os.startfile(os.path.abspath(RECORDINGS_DIR)))
    btn_open_rec.pack(anchor="w", padx=10, pady=(15, 10))

    def save_settings():
        app_config["gemini_api_key"] = entry_api.get().strip()
        save_config(app_config)
        load_all_categorized_context()
        win.destroy()

    ctk.CTkButton(win, text="Save & Sync", fg_color="#238636", command=save_settings).pack(pady=10)

def create_ui_components(root, transcriber, speaker_queue, mic_queue):
    ctk.set_appearance_mode("dark")
    root.title("Abujami Assistant (Dual Engine)")
    root.geometry("940x500+180+40")
    root.attributes("-topmost", True)
    root.attributes("-alpha", app_config.get("alpha", 0.82))
    root.configure(fg_color="#0d1117")

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)

    top_bar = ctk.CTkFrame(root, height=38, fg_color="#161b22")
    top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=6)

    # محول الأوضاع في الواجهة
    mode_selector = ctk.CTkSegmentedButton(
        top_bar, values=["Interview", "Meeting"],
        command=lambda m: toggle_mode(m, btn_action, transcript_box, answer_box, status_lbl, transcriber)
    )
    mode_selector.set("Interview")
    mode_selector.pack(side="left", padx=8, pady=5)

    btn_action = ctk.CTkButton(
        top_bar, text=f"Answer ({app_config.get('hotkey', 'alt+q')})", width=120, height=26,
        font=("Segoe UI", 11, "bold"), fg_color="#1f6feb",
        command=lambda: handle_main_action(transcriber, btn_action, status_lbl, transcript_box, answer_box)
    )
    btn_action.pack(side="left", padx=5)

    btn_clear = ctk.CTkButton(
        top_bar, text="Clear", width=55, height=26, fg_color="#21262d", hover_color="#da3633",
        command=lambda: clear_context(transcriber, speaker_queue, mic_queue, transcript_box, answer_box, status_lbl)
    )
    btn_clear.pack(side="left", padx=5)

    btn_settings = ctk.CTkButton(
        top_bar, text="⚙ Settings", width=75, height=26, fg_color="#21262d",
        command=lambda: open_settings_window(root, transcriber, answer_box, stealth_badge)
    )
    btn_settings.pack(side="left", padx=5)

    mic_bar = ctk.CTkProgressBar(top_bar, width=40, height=8, progress_color="#39ff14")
    mic_bar.set(0)
    mic_bar.pack(side="left", padx=6)

    spk_bar = ctk.CTkProgressBar(top_bar, width=40, height=8, progress_color="#58a6ff")
    spk_bar.set(0)
    spk_bar.pack(side="left", padx=6)

    stealth_badge = ctk.CTkLabel(top_bar, text="🛡️ Stealth: ON", font=("Segoe UI", 11, "bold"), text_color="#39ff14")
    stealth_badge.pack(side="left", padx=10)

    status_lbl = ctk.CTkLabel(top_bar, text="Mode: Interview Copilot", font=("Segoe UI", 11), text_color="#7ee787")
    status_lbl.pack(side="right", padx=10)

    # شاشة الاستماع
    f1 = ctk.CTkFrame(root, fg_color="#161b22")
    f1.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
    transcript_box = ctk.CTkTextbox(f1, font=("Consolas", 12), text_color="#c9d1d9", fg_color="transparent", wrap="word")
    transcript_box.pack(fill="both", expand=True, padx=6, pady=6)

    # شاشة الإجابة
    f2 = ctk.CTkFrame(root, fg_color="#161b22")
    f2.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
    answer_box = ctk.CTkTextbox(f2, font=("Segoe UI", 13), text_color="#f0f6fc", fg_color="transparent", wrap="word")
    answer_box.pack(fill="both", expand=True, padx=8, pady=8)
    write_in_textbox(answer_box, "★ Interview Mode Active: Press [Alt + Q] or click 'Answer' for instant talking points.")

    root.bind('<Return>', lambda e: handle_main_action(transcriber, btn_action, status_lbl, transcript_box, answer_box))
    try: keyboard.add_hotkey(app_config.get('hotkey', 'alt+q'), lambda: handle_main_action(transcriber, btn_action, status_lbl, transcript_box, answer_box))
    except: pass

    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    set_stealth_mode(hwnd, True)

    return transcript_box, answer_box, mic_bar, spk_bar, status_lbl

def main():
    try: subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("ERROR: ffmpeg is missing.")
        return

    root = ctk.CTk()
    speaker_queue = queue.Queue()
    mic_queue = queue.Queue()

    user_audio_recorder = AudioRecorder.DefaultMicRecorder()
    user_audio_recorder.record_into_queue(mic_queue)
    time.sleep(0.5)

    speaker_audio_recorder = AudioRecorder.DefaultSpeakerRecorder()
    speaker_audio_recorder.record_into_queue(speaker_queue)

    model = TranscriberModels.get_model('--api' in sys.argv)
    transcriber = AudioTranscriber(user_audio_recorder.source, speaker_audio_recorder.source, model)
    threading.Thread(target=transcriber.transcribe_audio_queue, args=(speaker_queue, mic_queue), daemon=True).start()

    transcript_box, answer_box, mic_bar, spk_bar, status_lbl = create_ui_components(root, transcriber, speaker_queue, mic_queue)
    update_ui_loop(transcriber, transcript_box, answer_box, mic_bar, spk_bar, status_lbl)
    root.mainloop()

if __name__ == "__main__":
    main()