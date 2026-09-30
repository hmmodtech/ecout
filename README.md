# 🎧 Abujami Assistant (ecout)

Real-time interview copilot for Windows, based on [SevaSk/ecoute](https://github.com/SevaSk/ecoute).

It listens to **your speakers (the interviewer)** and **your microphone (you)**, transcribes both
locally with Whisper, detects the interviewer's question the moment it is spoken and shows an answer:

1. **📚 QA file first** – your own question/answer bank in `context/`
2. **🔥 HEAT file second** – your training material (and any other reference docs)
3. **✨ Gemini last** – only when the files do not contain a confident answer
   (it is still given your files + CV as context, and streams the reply word by word)

> Arabic summary: see [العربية](#العربية) at the bottom.

---

## What changed vs. the original fork

| Problem | Fix |
|---|---|
| Audio stopped when the headset / default device changed | New `audio_engine.py`: polls the Windows default endpoints (pycaw) and **re-opens the streams automatically** on change, stall or error. Manual **Reconnect** button too. |
| Slow: question caught late, answer arrived late | Local answers come from an in-memory search index (milliseconds). Whisper runs with `beam_size=1`, int8, VAD, a 2-worker queue that prioritises the interviewer's final phrases, plus interim transcripts every 2.5 s. Gemini uses the new `google-genai` SDK with **`thinking_level=minimal`** (3.x Flash models think at *medium* by default – the main source of latency and truncated answers). |
| Answers came only from the AI prompt | `knowledge_base.py` parses your QA/HEAT files into pairs & passages, `retrieval.py` (BM25 + fuzzy + optional Gemini embeddings) ranks them, `answer_engine.py` enforces **QA → HEAT → Gemini**. |
| `custom_speech_recognition` used `audioop`/`distutils` (removed in Python 3.12/3.13) | Replaced by direct `pyaudiowpatch` callback streams; **no FFmpeg needed**. |

## 📋 Prerequisites

* Windows 10/11, Python 3.10 – 3.13
* A Gemini API key (only for the AI fallback / semantic search)

## 🔧 Installation

```bat
git clone https://github.com/hmmodtech/ecout
cd ecout
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put your API key in **one** of these (never in git):

* `config.local.json` → `{"gemini_api_key": "AIza..."}` (created automatically when you save it in Settings), or
* environment variable `GEMINI_API_KEY`, or
* `keys.py` → `GEMINI_API_KEY = "AIza..."`

## 📁 Your files (`context/`)

Copy your documents into the `context/` folder (`.docx`, `.pdf`, `.txt`, `.md`):

| Content | Name it like… | Bucket |
|---|---|---|
| Interview questions & answers | `QA_*.docx`, `Interview Questions.docx`, `Q&A…` | **QA** (searched first) |
| HEAT / training notes | `HEAT_*.pdf`, `Hostile Environment Training…` | **HEAT** (second) |
| CV / résumé | `CV_*.pdf`, `Abujami_CV.docx` | CV (context for Gemini) |
| Job description | `JD_*.docx` | JD |
| Anything else | `extra_*` or any other name | EXTRA (searched with HEAT) |

The QA parser understands `Q:`/`A:`, `Q1.`/`Answer:`, `Question:`/`Answer:`, `Interviewer:`/`Candidate:`,
Arabic `س:`/`ج:`, numbered lists, questions followed by a paragraph, tables in Word, etc.
**Verify what was parsed** (do this once!):

```bat
python knowledge_base.py context
```

You can also add files from the app: **Settings → Upload QA / Upload HEAT**.

## ▶ Running

```bat
python main.py          # or double-click run.bat
python main.py --api    # use a cloud ASR instead of local Whisper (OpenAI whisper-1 or Gemini)
```

* **Interview mode** – auto-answers every question the *Speaker* asks. `Alt+Q` (configurable) or the
  **Answer** button forces an answer for the last thing said.
* **Meeting mode** – records a transcript and exports it to Word.
* Status bar shows the current mic/speaker device names, ASR latency and knowledge-base counts.

## ⚙ Settings worth knowing (`config.json`)

| Key | Default | Meaning |
|---|---|---|
| `auto_answer` | `true` | Answer speaker questions automatically |
| `qa_high_confidence` / `qa_medium_confidence` | `0.62` / `0.40` | Above *high* → answer straight from the file; between → shown as "closest match" and Gemini is asked; below → Gemini |
| `semantic_search` | `true` | Also match by meaning using Gemini embeddings (cached in `context/.qa_embeddings_*.json`) |
| `ai_refine_local_answers` | `false` | Ask Gemini to polish a file answer (slower) |
| `whisper_model` | `base.en` | `tiny.en` = fastest, `small.en` = more accurate (GPU recommended) |
| `asr_beam_size` / `asr_workers` | `1` / `2` | Speed knobs for Whisper |
| `pause_threshold` | `0.5` s | Silence that ends a phrase |
| `interim_interval` | `2.5` s | How often long phrases produce partial transcripts |
| `device_role` | `auto` | Which Windows default device to follow: `auto`, `multimedia`, `communications` |
| `transcribe_mic` | `true` | Turn off to save CPU if you only need the interviewer |
| `model_name` / `llm_thinking` | `gemini-3.6-flash` / `minimal` | Gemini model and thinking level (`minimal`, `low`, `medium`, `high`, `default`) |

## 🧪 Tests

```bat
pip install -r requirements-dev.txt
pytest tests -q
```

The suite (70 tests) covers the parser, retrieval, question detection, the answer pipeline,
the audio engine (with a fake PyAudio) and the transcriber queue – no audio hardware needed.

---

## العربية

**Abujami Assistant** – مساعد مقابلات فوري على ويندوز.

* يسمع صوت **السماعة (المحاور)** و**المايك (أنت)** ويحوّلهما لنص محلياً بـ Whisper.
* يلتقط السؤال لحظة نطقه ويبحث عن الجواب بهذا الترتيب: **ملف الأسئلة (QA) ← ملف HEAT ← Gemini**.
* لو تغيّر جهاز الصوت (سماعة/مايك) يعيد الاتصال تلقائياً بالجهاز الافتراضي الحالي، وفيه زر **Reconnect** يدوي.

### التشغيل
1. `pip install -r requirements.txt`
2. ضع ملفاتك في مجلد `context/`:
   * ملف الأسئلة: اسمه يحتوي `QA` أو `Questions` أو `Interview` (مثل `QA_Interview.docx`)
   * ملف التدريب: اسمه يحتوي `HEAT` أو `Training`
   * السيرة الذاتية: `CV_*.pdf`
3. تأكد من قراءة الملفات: `python knowledge_base.py context`
4. ضع مفتاح Gemini في `config.local.json` أو متغير البيئة `GEMINI_API_KEY` (لا تضعه في `config.json`).
5. `python main.py`

### إعدادات مهمة
* `auto_answer`: الرد التلقائي على أسئلة المحاور.
* `whisper_model`: ‏`tiny.en` أسرع، `base.en` متوازن، `small.en` أدق.
* `device_role`: ‏`auto` يتابع جهاز الاتصالات الافتراضي في ويندوز (المناسب لـ Teams/Zoom).
* `llm_thinking`: ‏`minimal` لأسرع رد من Gemini.
