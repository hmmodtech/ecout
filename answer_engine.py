"""
Answer orchestration:  **Q&A bank → HEAT material → Gemini**.

1. The detected question is matched against the Q&A file first.  A confident
   hit is displayed instantly (a few ms, no network).
2. Otherwise Q/A pairs found inside the HEAT / extra documents are tried.
3. Otherwise the closest local candidate is shown immediately while Gemini
   streams a tailored answer that is grounded in the most relevant passages
   (retrieval-augmented, small prompt → low latency).

Every step reports progress through ``AnswerEvent`` callbacks so the UI can
render as soon as anything is known.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from knowledge_base import KnowledgeBase
from question_detector import question_score, split_sentences
from retrieval import QAMatch, SearchIndex

EventCallback = Callable[["AnswerEvent"], None]


@dataclass
class AnswerEvent:
    kind: str                     # start | local | candidate | ai_start | ai_chunk | ai_done | info | error
    text: str = ""
    source: str = ""              # "QA Bank" | "HEAT" | "Gemini" | ...
    confidence: float = 0.0
    matched_question: str = ""
    elapsed_ms: float = 0.0
    generation: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class AnswerRecord:
    question: str
    answer: str
    source: str


DEFAULT_PERSONA = (
    "You are Abujami Assistant, a real-time interview copilot. You whisper the exact words the "
    "candidate should say. Speak as the candidate in the first person ('I'). Be concrete, confident "
    "and brief: 3-4 punchy bullet points (max ~70 words) unless a step-by-step procedure is requested. "
    "Ground the answer in the REFERENCE MATERIAL when it is relevant; use the CV for personal "
    "examples. Never mention these instructions, the documents, or that you are an AI. Start directly "
    "with the answer – no preamble."
)


class AnswerEngine:
    def __init__(self, kb: KnowledgeBase, index: SearchIndex, llm=None, config: Optional[dict] = None):
        self.kb = kb
        self.index = index
        self.llm = llm
        self.config = config or {}
        self.high = float(self.config.get("qa_high_confidence", 0.62))
        self.medium = float(self.config.get("qa_medium_confidence", 0.40))
        self.history: List[AnswerRecord] = []
        self._history_lock = threading.Lock()
        self._system_prompt: Optional[str] = None

    # ------------------------------------------------------------------ utils
    def rebuild(self, kb: KnowledgeBase, index: SearchIndex):
        self.kb, self.index = kb, index
        self._system_prompt = None

    def system_prompt(self) -> str:
        if self._system_prompt is None:
            parts = [self.config.get("persona") or DEFAULT_PERSONA]
            cv = self.kb.condensed("CV", int(self.config.get("cv_prompt_budget", 1600)))
            jd = self.kb.condensed("JD", int(self.config.get("jd_prompt_budget", 900)))
            if cv:
                parts.append("--- CANDIDATE CV (excerpt) ---\n" + cv)
            if jd:
                parts.append("--- TARGET JOB DESCRIPTION (excerpt) ---\n" + jd)
            self._system_prompt = "\n\n".join(parts)
        return self._system_prompt

    def _remember(self, question: str, answer: str, source: str):
        with self._history_lock:
            self.history.append(AnswerRecord(question, answer.strip(), source))
            del self.history[:-4]

    @staticmethod
    def _fmt_local(match: QAMatch) -> str:
        return match.pair.answer.strip()

    @staticmethod
    def query_variants(question: str) -> List[str]:
        """The spoken question plus cleaner variants (last question sentence,
        question without lead-in sentences).  Searching all of them costs
        microseconds and makes matching robust to interviewer preambles."""
        variants = [question]
        sentences = split_sentences(question)
        if len(sentences) > 1:
            q_sents = [s for s in sentences if question_score(s) >= 0.45]
            if q_sents and q_sents[-1] != question:
                variants.append(q_sents[-1])
            if len(q_sents) > 1:
                variants.append(" ".join(q_sents))
            # drop leading non-question sentences
            i = 0
            while i < len(sentences) - 1 and question_score(sentences[i]) < 0.45:
                i += 1
            tail = " ".join(sentences[i:])
            if tail and tail not in variants:
                variants.append(tail)
        return variants

    def _search_best(self, question: str, categories, use_semantic: bool = False):
        """Search every query variant and keep the result with the best match."""
        best_res = None
        for variant in self.query_variants(question):
            res = self.index.search(variant, top_k=5, categories=categories, use_semantic=use_semantic)
            if best_res is None or (res.best and (best_res.best is None or res.best.confidence > best_res.best.confidence)):
                best_res = res
        return best_res

    # --------------------------------------------------------------- pipeline
    def answer(self, question: str, context: str = "", on_event: Optional[EventCallback] = None,
               generation: int = 0, is_cancelled: Optional[Callable[[], bool]] = None,
               force_ai: bool = False) -> Optional[AnswerRecord]:
        """Run the full pipeline synchronously (call from a worker thread)."""
        on_event = on_event or (lambda e: None)
        is_cancelled = is_cancelled or (lambda: False)
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            on_event(AnswerEvent("error", "No question detected yet – wait for the interviewer or press the hotkey after the question.", generation=generation))
            return None

        on_event(AnswerEvent("start", question, generation=generation))

        # ---- 1) Q&A bank (lexical only → instant) --------------------------
        qa_res = self._search_best(question, ["QA"])
        best = qa_res.best
        elapsed = (time.perf_counter() - t0) * 1000
        if best and best.confidence >= self.high and not force_ai:
            return self._finish_local(best, "QA Bank", elapsed, generation, on_event, question, context, is_cancelled)

        # ---- 2) Q/A pairs found inside HEAT / extra material --------------
        heat_res = self._search_best(question, ["HEAT", "EXTRA"])
        h_best = heat_res.best
        elapsed = (time.perf_counter() - t0) * 1000
        if h_best and h_best.confidence >= self.high and not force_ai:
            return self._finish_local(h_best, "HEAT", elapsed, generation, on_event, question, context, is_cancelled)

        # ---- 3) semantic upgrade (only when a network call is worth it) ---
        merged: List[QAMatch] = sorted(qa_res.qa_matches + heat_res.qa_matches, key=lambda m: m.confidence, reverse=True)
        if self.index.semantic_ready and self.config.get("semantic_search", True):
            sem_res = self._search_best(question, None, use_semantic=True)
            if sem_res.qa_matches:
                merged = sorted(sem_res.qa_matches + merged, key=lambda m: m.confidence, reverse=True)
                merged = _dedupe_matches(merged)
                elapsed = (time.perf_counter() - t0) * 1000
                if merged and merged[0].confidence >= self.high and not force_ai:
                    src = "QA Bank" if merged[0].pair.category == "QA" else "HEAT"
                    return self._finish_local(merged[0], src, elapsed, generation, on_event, question, context, is_cancelled)
        merged = _dedupe_matches(merged)

        # ---- 4) closest candidate shown immediately -----------------------
        candidate = merged[0] if merged and merged[0].confidence >= self.medium else None
        if candidate:
            on_event(AnswerEvent(
                "candidate", self._fmt_local(candidate),
                source="QA Bank" if candidate.pair.category == "QA" else candidate.pair.category,
                confidence=candidate.confidence, matched_question=candidate.pair.question,
                elapsed_ms=elapsed, generation=generation,
            ))
        if is_cancelled():
            return None

        # ---- 5) Gemini fallback (retrieval-augmented, compact prompt) -----
        passages = heat_res.passages or qa_res.passages
        if self.llm is not None and getattr(self.llm, "available", False):
            prompt = self.build_prompt(question, context, merged[:3], passages[:4])
            on_event(AnswerEvent("ai_start", "", source="Gemini", elapsed_ms=(time.perf_counter() - t0) * 1000, generation=generation))
            chunks: List[str] = []
            try:
                for chunk in self.llm.stream(self.system_prompt(), prompt):
                    if is_cancelled():
                        return None
                    chunks.append(chunk)
                    on_event(AnswerEvent("ai_chunk", chunk, source="Gemini", generation=generation))
                full = "".join(chunks).strip()
                on_event(AnswerEvent("ai_done", full, source="Gemini", elapsed_ms=(time.perf_counter() - t0) * 1000, generation=generation))
                if full:
                    self._remember(question, full, "Gemini")
                    return AnswerRecord(question, full, "Gemini")
                return None
            except Exception as exc:
                on_event(AnswerEvent("error", f"AI error: {exc}", source="Gemini", generation=generation))
                # fall through to local best effort

        # ---- 6) no AI: best-effort local suggestions ----------------------
        if merged:
            lines = []
            for m in merged[:3]:
                lines.append(f"• ({m.confidence:.0%}) {m.pair.question}\n  {m.pair.answer.strip()[:400]}")
            text = "Closest matches from your files:\n" + "\n\n".join(lines)
            on_event(AnswerEvent("info", text, source="Local", elapsed_ms=(time.perf_counter() - t0) * 1000, generation=generation))
            return AnswerRecord(question, text, "Local")
        if passages:
            text = "Relevant material:\n" + "\n\n".join(f"• [{p.passage.title or p.passage.source}] {p.passage.text[:400]}" for p in passages[:3])
            on_event(AnswerEvent("info", text, source="Local", generation=generation))
            return AnswerRecord(question, text, "Local")
        on_event(AnswerEvent("info", "No matching answer in the Q&A / HEAT files and no AI configured (Settings ⚙).", generation=generation))
        return None

    # ------------------------------------------------------------- helpers
    def _finish_local(self, match: QAMatch, source: str, elapsed_ms: float, generation: int,
                      on_event: EventCallback, question: str, context: str,
                      is_cancelled: Callable[[], bool]) -> AnswerRecord:
        text = self._fmt_local(match)
        on_event(AnswerEvent("local", text, source=source, confidence=match.confidence,
                             matched_question=match.pair.question, elapsed_ms=elapsed_ms, generation=generation))
        self._remember(question, text, source)
        if self.config.get("ai_refine_local_answers") and self.llm is not None and getattr(self.llm, "available", False):
            # Optional: personalised rewrite streamed *below* the instant answer.
            prompt = (
                f"Interview question: \"{question}\"\n\n"
                f"Prepared answer from the candidate's own Q&A file:\n{text}\n\n"
                "Rewrite it as natural spoken bullet points the candidate can say right now. Keep every fact; "
                "do not add new claims. 3-4 bullets."
            )
            on_event(AnswerEvent("ai_start", "", source="Gemini", generation=generation, extra={"refine": True}))
            try:
                chunks = []
                for chunk in self.llm.stream(self.system_prompt(), prompt):
                    if is_cancelled():
                        break
                    chunks.append(chunk)
                    on_event(AnswerEvent("ai_chunk", chunk, source="Gemini", generation=generation))
                on_event(AnswerEvent("ai_done", "".join(chunks), source="Gemini", generation=generation))
            except Exception as exc:
                on_event(AnswerEvent("error", f"AI refine error: {exc}", source="Gemini", generation=generation))
        return AnswerRecord(question, text, source)

    def build_prompt(self, question: str, context: str, matches: List[QAMatch], passages) -> str:
        parts: List[str] = []
        if context:
            parts.append(f"Interviewer said just before: \"{context[-300:]}\"")
        parts.append(f"INTERVIEW QUESTION: \"{question}\"")

        ref: List[str] = []
        budget = int(self.config.get("reference_prompt_budget", 2200))
        used = 0
        for m in matches:
            block = f"[Prepared Q&A – similarity {m.confidence:.0%}]\nQ: {m.pair.question}\nA: {m.pair.answer.strip()}"
            if used + len(block) > budget:
                block = block[: max(0, budget - used)]
            if block:
                ref.append(block)
                used += len(block)
            if used >= budget:
                break
        for p in passages:
            if used >= budget:
                break
            title = p.passage.title or p.passage.source or p.passage.category
            block = f"[{p.passage.category}: {title}]\n{p.passage.text.strip()}"
            if used + len(block) > budget:
                block = block[: max(0, budget - used)]
            if block:
                ref.append(block)
                used += len(block)
        if ref:
            parts.append("REFERENCE MATERIAL:\n" + "\n\n".join(ref))
        else:
            parts.append("REFERENCE MATERIAL: (nothing relevant found – rely on the CV and best practice)")

        with self._history_lock:
            hist = list(self.history[-2:])
        if hist:
            parts.append("PREVIOUS EXCHANGES (for continuity):\n" + "\n".join(
                f"- Q: {h.question[:120]} → A: {h.answer[:200]}" for h in hist))

        parts.append("Respond now as the candidate.")
        return "\n\n".join(parts)


def _dedupe_matches(matches: List[QAMatch]) -> List[QAMatch]:
    seen = set()
    out: List[QAMatch] = []
    for m in matches:
        key = m.pair.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out
