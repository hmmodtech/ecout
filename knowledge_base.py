"""
Knowledge base loading for Abujami Assistant.

Responsibilities
----------------
* Extract text from the files dropped into the ``context/`` folder
  (.docx incl. tables, .pdf, .txt, .md).
* Categorise files (CV / JD / QA / HEAT / EXTRA) using the same naming
  conventions the settings window uses (``qa_*``, ``heat_*`` ...).
* Parse **question / answer pairs** out of the Q&A bank (and out of any
  other document that happens to contain Q/A style content).
* Split long training material (HEAT) into retrievable passages.

Everything in here is pure python so it can be unit tested without audio
hardware or a GUI.
"""
from __future__ import annotations

import glob
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # optional dependency – .docx support
    import docx  # type: ignore
    from docx.oxml.ns import qn  # type: ignore
    from docx.table import Table as _DocxTable  # type: ignore
    from docx.text.paragraph import Paragraph as _DocxParagraph  # type: ignore

    DOCX_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    DOCX_AVAILABLE = False

CONTEXT_DIR = "context"

# File name patterns per category.  Order matters: the first category whose
# pattern matches wins.  ``qa_bank`` is the name used by the settings window.
# File-name → category rules.  Matching is case-insensitive and works on any
# part of the name, so "Security Interview Questions.docx", "HEAT notes.pdf",
# "Abujami_CV.pdf" all land in the right bucket.  A leading prefix such as
# "qa_", "heat_", "cv_", "jd_", "extra_" always wins.
CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "QA": [r"^qa[\W_]", r"^q&a", r"^qna", r"(?:^|[\W_])qa(?:[\W_]|$)", r"q&a", r"qna", r"question", r"interview", r"faq",
           r"أسئلة", r"اسئلة", r"سؤال", r"مقابلة"],
    "HEAT": [r"^heat[\W_]", r"heat", r"hefat", r"hostile", r"training", r"تدريب", r"بيئة"],
    "CV": [r"^cv[\W_]", r"(?:^|[\W_])cv(?:[\W_]|$)", r"resume", r"résumé", r"curriculum", r"سيرة"],
    "JD": [r"^jd[\W_]", r"(?:^|[\W_])jd(?:[\W_]|$)", r"job", r"vacancy", r"role", r"وظيف"],
    "EXTRA": [r".*"],  # everything else
}
FORCED_PREFIXES = {"qa_": "QA", "qa-": "QA", "heat_": "HEAT", "heat-": "HEAT", "cv_": "CV", "cv-": "CV",
                   "jd_": "JD", "jd-": "JD", "extra_": "EXTRA", "extra-": "EXTRA", "ref_": "EXTRA", "doc_": "EXTRA"}

SUPPORTED_EXTENSIONS = (".docx", ".pdf", ".txt", ".md")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class QAPair:
    question: str
    answer: str
    source: str = ""
    category: str = "QA"

    def short(self, n: int = 80) -> str:
        q = self.question if len(self.question) <= n else self.question[: n - 1] + "…"
        return q


@dataclass
class Passage:
    text: str
    title: str = ""
    source: str = ""
    category: str = "HEAT"


@dataclass
class KnowledgeStats:
    files: Dict[str, List[str]] = field(default_factory=dict)
    qa_pairs: Dict[str, int] = field(default_factory=dict)
    passages: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        for cat in ("QA", "HEAT", "CV", "JD", "EXTRA"):
            n_files = len(self.files.get(cat, []))
            if not n_files:
                continue
            bits = [f"{n_files} file{'s' if n_files != 1 else ''}"]
            if self.qa_pairs.get(cat):
                bits.append(f"{self.qa_pairs[cat]} Q/A")
            if self.passages.get(cat):
                bits.append(f"{self.passages[cat]} passages")
            parts.append(f"{cat}: " + ", ".join(bits))
        return " · ".join(parts) if parts else "no context files"


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
_QUESTION_HEADER_WORDS = ("question", "questions", "q", "سؤال", "السؤال", "الأسئلة")
_ANSWER_HEADER_WORDS = ("answer", "answers", "a", "response", "model answer", "جواب", "الجواب", "الإجابة", "الاجابة")


def _iter_docx_blocks(document):
    """Yield paragraphs and tables of a python-docx document in body order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield _DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield _DocxTable(child, document)


def _docx_table_to_text(table) -> str:
    """Render a docx table.  Two-column question/answer tables are rendered
    as ``Q:``/``A:`` lines so the Q&A parser can pick them up."""
    rows: List[List[str]] = []
    for row in table.rows:
        cells = []
        seen = set()
        for cell in row.cells:
            # merged cells are repeated by python-docx – skip duplicates
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(p.text.strip() for p in cell.paragraphs if p.text.strip()))
        if any(c for c in cells):
            rows.append(cells)
    if not rows:
        return ""

    header = [c.strip().lower() for c in rows[0]]
    q_idx = next((i for i, h in enumerate(header) if h in _QUESTION_HEADER_WORDS), None)
    a_idx = next((i for i, h in enumerate(header) if h in _ANSWER_HEADER_WORDS), None)
    data_rows = rows
    if q_idx is not None and a_idx is not None:
        data_rows = rows[1:]
    elif len(header) >= 2 and _looks_like_question(rows[0][0]) and not _looks_like_question(rows[0][1] if len(rows[0]) > 1 else ""):
        # No header, but the first column already looks like questions.
        q_idx, a_idx = 0, 1

    lines: List[str] = []
    if q_idx is not None and a_idx is not None:
        for cells in data_rows:
            if len(cells) <= max(q_idx, a_idx):
                continue
            q, a = cells[q_idx].strip(), cells[a_idx].strip()
            if q and a:
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")
                lines.append("")
    else:
        for cells in rows:
            lines.append(" | ".join(c for c in cells if c))
    return "\n".join(lines)


def extract_text_from_file(file_path: str) -> str:
    """Return the plain text of a .docx / .pdf / .txt / .md file ("" on error)."""
    if not file_path or not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".docx":
            if not DOCX_AVAILABLE:
                return ""
            document = docx.Document(file_path)
            chunks: List[str] = []
            for block in _iter_docx_blocks(document):
                if isinstance(block, _DocxParagraph):
                    t = block.text.strip()
                    if not t:
                        chunks.append("")
                        continue
                    style = (block.style.name or "").lower() if block.style is not None else ""
                    if style.startswith("heading") or style == "title":
                        level = 2
                        m = re.search(r"(\d+)", style)
                        if m:
                            level = min(int(m.group(1)) + 1, 6)
                        chunks.append("#" * level + " " + t)
                    elif style.startswith("list"):
                        chunks.append("- " + t)
                    else:
                        chunks.append(t)
                else:
                    chunks.append(_docx_table_to_text(block))
            text = "\n".join(chunks)
        elif ext == ".pdf":
            from pypdf import PdfReader  # local import: optional dependency

            reader = PdfReader(file_path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[KnowledgeBase] Could not read {file_path}: {exc}")
        return ""
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Q&A parsing
# --------------------------------------------------------------------------- #
_Q_LABEL = re.compile(
    r"^\s*[\*\_#>\-•·]*\s*"
    r"(?:q|qn|question|interviewer|int|panel|سؤال|السؤال|س)\s*"
    r"(?:no\.?|#)?\s*\d{0,3}\s*[\.\)\-:：]?\s*[:：\-–—.)]\s*(.*)$",
    re.IGNORECASE,
)
_A_LABEL = re.compile(
    r"^\s*[\*\_#>\-•·]*\s*"
    r"(?:a|ans|answer|model answer|suggested answer|sample answer|response|candidate|me|جواب|الجواب|الإجابة|الاجابة|ج)\s*"
    r"\d{0,3}\s*[\.\)\-:：]?\s*[:：\-–—.)]\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERING = re.compile(r"^\s*(?:\(?\d{1,3}[\.\)\-:]|[a-zA-Z][\.\)]|[•\-\*·►▪]|#{1,6})\s+")
_QUESTION_WORDS = (
    "what", "why", "how", "when", "where", "who", "whom", "which", "whose",
    "can", "could", "would", "will", "should", "shall", "do", "does", "did",
    "is", "are", "was", "were", "have", "has", "had", "tell me", "tell us",
    "describe", "explain", "walk me", "walk us", "give me", "give us", "share",
    "talk about", "elaborate", "define", "name", "list", "imagine", "suppose",
    "if you", "in your", "any experience", "are you", "please describe",
    "ما", "ماذا", "لماذا", "كيف", "متى", "أين", "اين", "من", "هل", "أي", "اي",
    "حدثني", "اشرح", "صف", "اذكر", "وش", "شو", "ليش", "كيفية",
)


def _looks_like_question(line: str) -> bool:
    if not line:
        return False
    s = _NUMBERING.sub("", line).strip().strip("*_").strip()
    if not s or len(s) > 400:
        return False
    if s.endswith("?") or s.endswith("؟"):
        return True
    low = s.lower()
    return any(low.startswith(w + " ") or low.startswith(w + ",") for w in _QUESTION_WORDS) and len(s.split()) >= 3


_STRONG_STARTERS = (
    "describe", "tell me", "tell us", "explain", "give me", "give us", "walk me", "walk us", "talk about",
    "share", "discuss", "what", "why", "how", "when", "where", "who", "which", "can you", "could you",
    "would you", "have you", "do you", "did you", "are you", "is there", "in your", "imagine", "suppose",
    "حدثني", "اشرح", "صف", "اذكر", "ما", "ماذا", "لماذا", "كيف", "متى", "أين", "هل",
)


def _is_strong_question(line: str) -> bool:
    """Numbered lines without a '?' still count as questions when they start
    like an interview prompt ("3. Describe a time you ...")."""
    s = _NUMBERING.sub("", line).strip().strip("*_").strip().lower()
    if len(s.split()) < 5 or len(s) > 300:
        return False
    return any(s.startswith(w + " ") for w in _STRONG_STARTERS)


def _clean_question(q: str) -> str:
    q = _NUMBERING.sub("", q).strip()
    q = q.strip("*_ ").strip()
    return q


def _clean_answer(a: str) -> str:
    a = a.strip()
    a = re.sub(r"\n{3,}", "\n\n", a)
    return a.strip()


def _dedupe(pairs: List[QAPair]) -> List[QAPair]:
    seen = set()
    out: List[QAPair] = []
    for p in pairs:
        key = re.sub(r"\W+", " ", p.question.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _parse_labeled(lines: List[str], source: str, category: str) -> List[QAPair]:
    pairs: List[QAPair] = []
    q_buf: List[str] = []
    a_buf: List[str] = []
    state = None  # None | "q" | "a"
    q_complete = False

    def flush():
        nonlocal q_buf, a_buf
        q = _clean_question(" ".join(x.strip() for x in q_buf if x.strip()))
        a = _clean_answer("\n".join(a_buf))
        if q and a:
            pairs.append(QAPair(q, a, source, category))
        q_buf, a_buf = [], []

    def question_is_complete(text: str) -> bool:
        t = text.strip()
        return bool(t) and (t.endswith(("?", "؟", ".", "!")) or len(t.split()) >= 4)

    for raw in lines:
        line = raw.rstrip()
        mq = _Q_LABEL.match(line)
        ma = _A_LABEL.match(line) if not mq else None
        if mq:
            if q_buf and a_buf:
                flush()
            elif state == "q" and q_buf:
                # a question without an answer: keep the newest one only
                q_buf = []
            state = "q"
            q_buf.append(mq.group(1))
            q_complete = question_is_complete(mq.group(1))
        elif ma:
            state = "a"
            a_buf.append(ma.group(1))
        else:
            if state == "q":
                if not line.strip():
                    if q_buf and q_complete:
                        # blank line after a complete question → answer follows (implicit "A:")
                        state = "a"
                    continue
                if q_complete:
                    state = "a"
                    a_buf.append(line)
                else:
                    q_buf.append(line)
                    q_complete = True
            elif state == "a":
                a_buf.append(line)
    if q_buf and a_buf:
        flush()
    return pairs


def _parse_question_lines(lines: List[str], source: str, category: str) -> List[QAPair]:
    """Fallback: a line that looks like a question followed by answer lines."""
    pairs: List[QAPair] = []
    q: Optional[str] = None
    a_buf: List[str] = []

    def flush():
        nonlocal q, a_buf
        if q:
            a = _clean_answer("\n".join(a_buf))
            if a:
                pairs.append(QAPair(_clean_question(q), a, source, category))
        q, a_buf = None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            if q and a_buf:
                a_buf.append("")
            continue
        mq = _Q_LABEL.match(line)
        ma = _A_LABEL.match(line) if not mq else None
        if mq:
            line = mq.group(1).strip()
            if not line:
                continue
        elif ma:
            if q is not None:
                a_buf.append(ma.group(1))
            continue
        is_heading_q = line.startswith("#") and _looks_like_question(line.lstrip("# "))
        numbered_q = bool(_NUMBERING.match(line)) and not line.startswith("#") and _is_strong_question(line)
        if mq or is_heading_q or numbered_q or (_looks_like_question(line) and (line.endswith("?") or line.endswith("؟") or line.startswith("#"))):
            flush()
            q = line.lstrip("# ").strip()
        elif q is not None:
            a_buf.append(_NUMBERING.sub("", line) if line.startswith(("#",)) else line)
    flush()
    return pairs


def parse_qa_pairs(text: str, source: str = "", category: str = "QA") -> List[QAPair]:
    """Parse question/answer pairs from free text.

    Supported layouts (mixed layouts inside one file are fine):

    * ``Q: ... / A: ...`` (also ``Question 1:``, ``Q1)``, ``Interviewer:`` /
      ``Candidate:``, Arabic ``س:`` / ``ج:``)
    * numbered / bulleted questions ending with ``?`` followed by the answer
      paragraph(s)
    * markdown headings that are questions followed by the answer body
    * two-column tables (rendered by :func:`extract_text_from_file`)
    """
    if not text or not text.strip():
        return []
    lines = normalize_text(text).split("\n")

    labeled = _parse_labeled(lines, source, category)
    if len(labeled) >= 2:
        # Some files mix labelled pairs with plain "question?" blocks.
        extra = [p for p in _parse_question_lines(lines, source, category)
                 if not any(p.question.lower() == l.question.lower() for l in labeled)]
        return _dedupe(labeled + extra)
    fallback = _parse_question_lines(lines, source, category)
    if len(fallback) >= 1:
        return _dedupe(labeled + fallback)
    return _dedupe(labeled)


# --------------------------------------------------------------------------- #
# Passage chunking (HEAT / EXTRA training material)
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|\d+(?:\.\d+)*[\.\)]?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9 &/\-]{4,80})$")
_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?؟])\s+(?=[A-Z\"'\(\[0-9\u0600-\u06FF])")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if s.endswith((".", "?", "!", ":", "؟", ",", ";")) and not s.startswith("#"):
        return False
    return bool(_HEADING_RE.match(s))


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_passages(text: str, source: str = "", category: str = "HEAT",
                   max_chars: int = 650, min_chars: int = 40) -> List[Passage]:
    """Split a document into passages of roughly ``max_chars`` characters,
    keeping the nearest heading as a title for context."""
    if not text or not text.strip():
        return []
    passages: List[Passage] = []
    title = ""
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        body = " ".join(buf).strip()
        if len(body) >= min_chars:
            passages.append(Passage(body, title, source, category))
        buf, buf_len = [], 0

    for para in re.split(r"\n\s*\n|\n(?=#)", normalize_text(text)):
        para = " ".join(l.strip() for l in para.split("\n") if l.strip())
        if not para:
            continue
        if _is_heading(para):
            flush()
            title = para.lstrip("# ").strip()
            continue
        for sent in split_sentences(para) or [para]:
            if buf_len + len(sent) > max_chars and buf:
                flush()
            # a single very long sentence gets its own passage
            buf.append(sent)
            buf_len += len(sent) + 1
        if buf_len >= max_chars * 0.6:
            flush()
    flush()
    return passages


# --------------------------------------------------------------------------- #
# Knowledge base container
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """Holds every piece of reference material in memory."""

    def __init__(self, context_dir: str = CONTEXT_DIR):
        self.context_dir = context_dir
        self.raw: Dict[str, str] = {c: "" for c in CATEGORY_PATTERNS}
        self.qa_pairs: List[QAPair] = []
        self.passages: List[Passage] = []
        self.stats = KnowledgeStats()

    # -- loading ------------------------------------------------------------
    @staticmethod
    def category_for(filename: str) -> str:
        """Pick the category bucket for a file name (case-insensitive)."""
        stem = os.path.splitext(os.path.basename(filename))[0].strip().lower()
        for prefix, cat in FORCED_PREFIXES.items():
            if stem.startswith(prefix):
                return cat
        for cat, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, stem, re.IGNORECASE):
                    return cat
        return "EXTRA"

    def categorize_files(self) -> Dict[str, List[str]]:
        os.makedirs(self.context_dir, exist_ok=True)
        assigned: Dict[str, List[str]] = {c: [] for c in CATEGORY_PATTERNS}
        for path in sorted(glob.glob(os.path.join(self.context_dir, "*"))):
            name = os.path.basename(path)
            if not os.path.isfile(path) or not name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            if name.startswith((".", "~$")):
                continue
            assigned[self.category_for(name)].append(path)
        return assigned

    def load(self) -> KnowledgeStats:
        files = self.categorize_files()
        raw: Dict[str, str] = {c: "" for c in CATEGORY_PATTERNS}
        qa_pairs: List[QAPair] = []
        passages: List[Passage] = []
        stats = KnowledgeStats(files=files)

        for cat, paths in files.items():
            texts = []
            for path in paths:
                content = extract_text_from_file(path)
                if not content:
                    continue
                name = os.path.basename(path)
                texts.append(f"[{cat} DOC: {name}]\n{content}")
                if cat in ("QA", "HEAT", "EXTRA"):
                    pairs = parse_qa_pairs(content, name, cat)
                    # For non-QA docs only trust the parser when it finds a real bank
                    if cat == "QA" or len(pairs) >= 3:
                        qa_pairs.extend(pairs)
                        stats.qa_pairs[cat] = stats.qa_pairs.get(cat, 0) + len(pairs)
                if cat in ("HEAT", "EXTRA", "JD"):
                    chunks = chunk_passages(content, name, cat)
                    passages.extend(chunks)
                    stats.passages[cat] = stats.passages.get(cat, 0) + len(chunks)
            raw[cat] = "\n\n".join(texts)

        self.raw = raw
        self.qa_pairs = _dedupe(qa_pairs)
        self.passages = passages
        self.stats = stats
        return stats

    # -- helpers ------------------------------------------------------------
    def pairs_by_category(self, category: str) -> List[QAPair]:
        return [p for p in self.qa_pairs if p.category == category]

    def vocabulary_hint(self, limit: int = 24) -> str:
        """Domain words (acronyms / capitalised terms) to bias the ASR model."""
        counter: Counter = Counter()
        text = " ".join([self.raw.get("QA", ""), self.raw.get("HEAT", ""), self.raw.get("JD", ""),
                         self.raw.get("EXTRA", "")])
        stop = {"the", "this", "that", "when", "what", "how", "why", "where", "doc", "question", "answer",
                "interviewer", "candidate", "yes", "no", "you", "your", "they", "there", "then", "also", "with"}
        for m in re.finditer(r"\b([A-Z]{2,6}s?|[A-Z][a-z]{3,}(?:\s[A-Z][a-z]{3,})?)\b", text):
            tok = m.group(1)
            if tok.lower() in stop:
                continue
            is_acronym = tok.isupper()
            if not is_acronym:
                # ignore ordinary sentence-initial capitalisation
                before = text[: m.start()].rstrip()
                if not before or before[-1] in ".!?:;\n]\"'“”•-–—":
                    continue
            counter[tok] += 1
        terms = [t for t, c in counter.most_common(limit * 2) if c >= 2 or (t.isupper() and len(t) >= 3)][:limit]
        return ", ".join(terms)

    def condensed(self, category: str, budget: int = 1500) -> str:
        """A budgeted excerpt of a category (used for LLM prompts)."""
        text = self.raw.get(category, "")
        if len(text) <= budget:
            return text
        return text[: budget - 1].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# CLI helper: python knowledge_base.py [context_dir]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys

    kb = KnowledgeBase(sys.argv[1] if len(sys.argv) > 1 else CONTEXT_DIR)
    st = kb.load()
    print("Files:")
    for cat, paths in st.files.items():
        for p in paths:
            print(f"  [{cat}] {os.path.basename(p)}")
    print("\nSummary:", st.summary())
    print(f"\nParsed {len(kb.qa_pairs)} Q/A pairs. First 5:")
    for pair in kb.qa_pairs[:5]:
        print(f"  Q: {pair.question}\n  A: {pair.answer[:160]}{'…' if len(pair.answer) > 160 else ''}\n")
    print(f"Passages: {len(kb.passages)}")
    for ps in kb.passages[:3]:
        print(f"  [{ps.title}] {ps.text[:120]}…")
    print("\nASR vocabulary hint:", kb.vocabulary_hint())
