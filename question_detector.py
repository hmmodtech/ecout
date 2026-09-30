"""
Detects interview questions in the live transcript and extracts the exact
question text to search for.

The transcript arrives as short phrases (one per pause).  A question may be
spread over several phrases ("Okay so next question. Tell me about a time…
when you had to deal with a hostile checkpoint?") so we look at a rolling
window of the interviewer's recent phrases and cut out the question-bearing
tail.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "whom", "which", "whose",
    "can", "could", "would", "will", "should", "shall", "do", "does", "did",
    "is", "are", "was", "were", "have", "has", "had", "am",
    "tell me", "tell us", "describe", "explain", "walk me", "walk us", "give me",
    "give us", "share", "talk about", "talk to me", "talk us", "elaborate", "define", "name",
    "list", "imagine", "suppose", "let's say", "lets say", "if you", "in your experience",
    "in your opinion", "in your own words", "any experience", "anything about",
    "so how", "so what", "so why", "so tell", "so can", "so could", "so would", "so do", "so if",
    "and how", "and what", "and why", "and tell", "and if", "and can", "and would",
    "now how", "now what", "now tell", "okay so", "okay tell", "ok so", "alright so",
    "first question", "next question", "last question", "final question", "another question",
    "my question", "the question", "question is", "i'd like to know", "i would like to know",
    "i want to know", "i'm curious", "i am curious", "curious to know", "wondering",
    "say you", "you said", "you mentioned", "picture this", "scenario",
    # Arabic
    "ما", "ماذا", "لماذا", "كيف", "متى", "أين", "اين", "من", "هل", "أي", "اي",
    "حدثني", "حدثنا", "اشرح", "صف", "اذكر", "وش", "شو", "ليش", "كيفية", "ممكن", "عرفني", "عرفنا",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?؟])\s+")
_TRAILING_FILLER = re.compile(r"\s*(?:yeah|yes|okay|ok|right|please|thanks|thank you|go ahead)[\.\?\!]*\s*$", re.IGNORECASE)
_STARTS_NON_QUESTION = re.compile(
    r"^(?:thank you|thanks|great|perfect|nice|good|excellent|awesome|okay|ok|alright|right|i see|got it|understood|"
    r"that's|thats|that is|makes sense|sounds good|mm|mhm|uh huh|yes|yeah|no|sure|cool|interesting)\b[\s,\.!]*$",
    re.IGNORECASE,
)


_FILLER_SENTENCE = re.compile(
    r"^(?:thanks?\b|thank you\b|welcome\b|hello\b|hi\b|good (?:morning|afternoon|evening)\b|nice to meet you\b|"
    r"great answer\b|perfect\b|excellent\b|that's (?:great|good|helpful|clear)\b|makes sense\b|"
    r"let's move on\b|moving on\b|next question\b|okay next\b|so next\b|alright next\b|my name is\b|i am the\b|i'm the\b)",
    re.IGNORECASE,
)


def _is_filler_sentence(text: str) -> bool:
    return bool(_FILLER_SENTENCE.match(text.strip()))


@dataclass
class DetectedQuestion:
    text: str            # the question itself (what we search for)
    context: str         # a bit of preceding interviewer speech
    score: float         # 0..1 how sure we are it is a question
    phrase_ids: Tuple[int, ...] = ()


def question_score(text: str) -> float:
    """Return a 0..1 score for how question-like ``text`` is."""
    if not text:
        return 0.0
    s = text.strip()
    if _STARTS_NON_QUESTION.match(s):
        return 0.0
    low = s.lower().lstrip("-–—•* ").strip()
    n_words = len(low.split())
    if n_words < 2:
        return 0.0

    score = 0.0
    if s.endswith("?") or s.endswith("؟"):
        score += 0.75
    elif "?" in s or "؟" in s:
        score += 0.45

    for starter in _QUESTION_STARTERS:
        if low.startswith(starter + " ") or low.startswith(starter + ",") or low == starter:
            score += 0.45 if len(starter) > 3 else 0.35
            break
    else:
        # question words in the middle ("... so what would you do?")
        if re.search(r"\b(?:what|why|how|when|where|who|which|would you|could you|can you|do you|did you|have you|are you|is there|tell me|describe|explain)\b", low):
            score += 0.25

    if n_words >= 4:
        score += 0.05
    if n_words > 60:
        score -= 0.15
    return max(0.0, min(1.0, score))


def is_question(text: str, threshold: float = 0.45) -> bool:
    return question_score(text) >= threshold


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_question(phrases: Sequence[Tuple[int, str, float]], now: Optional[float] = None,
                     window_seconds: float = 15.0, min_score: float = 0.45) -> Optional[DetectedQuestion]:
    """Find the most recent question in the interviewer's recent phrases.

    ``phrases`` is a chronologically ordered sequence of
    ``(phrase_id, text, end_timestamp)`` for the *Speaker* source.
    Returns ``None`` when the recent speech does not contain a question.
    """
    if not phrases:
        return None
    now = time.time() if now is None else now
    recent = [p for p in phrases if now - p[2] <= window_seconds] or [phrases[-1]]

    # Flatten into sentences while remembering which phrase they came from.
    sentences: List[Tuple[int, str]] = []
    for pid, text, _ in recent:
        for sent in split_sentences(text):
            sentences.append((pid, sent))
    if not sentences:
        return None

    # Walk backwards to find the last question sentence.
    last_q = None
    for idx in range(len(sentences) - 1, -1, -1):
        sc = question_score(sentences[idx][1])
        if sc >= min_score:
            last_q = idx
            break
    if last_q is None:
        return None

    # The question may span the preceding sentence(s) of the *same phrase*
    # when they are not complete statements (e.g. "Tell me about a time.
    # When you had to deal with an aggressive crowd?")
    start = last_q
    q_pid = sentences[last_q][0]
    while start - 1 >= 0 and sentences[start - 1][0] == q_pid:
        prev = sentences[start - 1][1]
        if _STARTS_NON_QUESTION.match(prev) or _is_filler_sentence(prev):
            break
        q_len = len(" ".join(s for _, s in sentences[start:last_q + 1]).split())
        if question_score(prev) >= 0.3 or (q_len <= 5 and len(prev.split()) <= 12):
            start -= 1
        else:
            break
    # Include immediately following short fragments of the same question
    # ("...checkpoint?" "Like, what would you do?")
    end = last_q
    while end + 1 < len(sentences) and sentences[end + 1][0] == q_pid:
        nxt = sentences[end + 1][1]
        if question_score(nxt) >= 0.3 or len(nxt.split()) <= 5:
            end += 1
        else:
            break

    q_text = " ".join(s for _, s in sentences[start:end + 1]).strip()
    q_text = _TRAILING_FILLER.sub("", q_text).strip() or q_text
    context = " ".join(s for _, s in sentences[max(0, start - 3):start]).strip()
    ids = tuple(sorted({pid for pid, _ in sentences[start:end + 1]}))
    return DetectedQuestion(text=q_text, context=context, score=question_score(q_text), phrase_ids=ids)


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF ]+", " ", text.lower()).strip()
