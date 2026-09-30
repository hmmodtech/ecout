import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_engine import AnswerEngine  # noqa: E402
from knowledge_base import KnowledgeBase, chunk_passages, parse_qa_pairs  # noqa: E402
from question_detector import extract_question, is_question  # noqa: E402
from retrieval import SearchIndex  # noqa: E402


# --------------------------------------------------------------------------- #
# question detection
# --------------------------------------------------------------------------- #
def test_question_score_positive_cases():
    for text in [
        "Tell me about yourself.",
        "How would you handle a hostile checkpoint?",
        "So what would you do if the convoy was ambushed",
        "Can you describe a time you dealt with an aggressive crowd?",
        "Okay so next question, why do you want to join us?",
        "Walk me through your medical training",
        "Any experience with kidnapping situations?",
        "حدثني عن نفسك",
        "كيف تتعامل مع نقطة تفتيش عدائية؟",
    ]:
        assert is_question(text), text


def test_question_score_negative_cases():
    for text in [
        "Thank you.",
        "Great, that makes sense.",
        "We are a humanitarian organisation working in twelve countries.",
        "Okay.",
        "The role is based in Juba with frequent travel to the field.",
    ]:
        assert not is_question(text), text


def test_extract_question_uses_last_question_in_window():
    now = 1000.0
    phrases = [
        (1, "Thanks for joining us today. We are a humanitarian organisation.", now - 40),
        (2, "Let me tell you a bit about the role first.", now - 30),
        (3, "It is based in Juba with travel to the field.", now - 12),
        (4, "So my first question. Tell me about a time when you had to deal with a hostile checkpoint?", now - 2),
    ]
    det = extract_question(phrases, now=now, window_seconds=15)
    assert det is not None
    assert det.text.startswith("So my first question.") or det.text.startswith("Tell me about a time")
    assert "hostile checkpoint" in det.text
    assert det.phrase_ids == (4,)


def test_extract_question_returns_none_without_question():
    now = 100.0
    phrases = [(1, "Thank you very much.", now - 1), (2, "That was very clear.", now - 0.5)]
    assert extract_question(phrases, now=now) is None


def test_extract_question_spanning_two_phrases_prefers_newest_question():
    now = 50.0
    phrases = [
        (1, "What is your biggest weakness?", now - 20),
        (2, "Interesting. And how do you prioritise when everything is urgent?", now - 1),
    ]
    det = extract_question(phrases, now=now, window_seconds=30)
    assert "prioritise" in det.text
    assert "weakness" not in det.text


# --------------------------------------------------------------------------- #
# answer engine
# --------------------------------------------------------------------------- #
QA_TEXT = """
Q: Tell me about yourself.
A: I am a security professional with eight years in hostile environments.
Q: How do you handle a hostile checkpoint?
A: Stay calm, keep hands visible, comply and report to ops afterwards.
Q: What are your salary expectations?
A: I am flexible and focused on the role.
"""

HEAT_TEXT = """# Kidnapping
The first 24 hours are the most dangerous. Stay calm, do not resist, humanise yourself.

Q: What is the golden rule during captivity?
A: Comply, stay calm, keep your dignity and look for proof-of-life opportunities.

# Medical
Catastrophic bleeding first: tourniquet high and tight, note the time, then airway and breathing.
"""


class FakeLLM:
    def __init__(self, available=True, reply="• I stay calm\n• I comply"):
        self.available = available
        self.reply = reply
        self.calls = []

    def stream(self, system_instruction, prompt, **kw):
        self.calls.append((system_instruction, prompt))
        for word in self.reply.split(" "):
            yield word + " "

    def describe(self):
        return "fake"


def build_engine(llm=None, config=None):
    kb = KnowledgeBase("does-not-exist")
    kb.qa_pairs = parse_qa_pairs(QA_TEXT, "qa.txt", "QA") + parse_qa_pairs(HEAT_TEXT, "heat.txt", "HEAT")
    kb.passages = chunk_passages(HEAT_TEXT, "heat.txt", "HEAT", max_chars=200)
    kb.raw["CV"] = "Jane Doe, Field Security Advisor, 8 years, HEAT certified."
    kb.raw["JD"] = "Security Officer – South Sudan."
    index = SearchIndex(kb.qa_pairs, kb.passages)
    return AnswerEngine(kb, index, llm, config or {}), kb, index


def collect(engine, question, **kw):
    events = []
    rec = engine.answer(question, on_event=events.append, **kw)
    return events, rec


def test_qa_bank_answers_instantly_without_llm():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm)
    events, rec = collect(engine, "Could you tell us a bit about yourself?")
    kinds = [e.kind for e in events]
    assert kinds == ["start", "local"]
    local = events[1]
    assert local.source == "QA Bank"
    assert local.confidence >= 0.62
    assert "eight years" in local.text
    assert local.elapsed_ms < 100
    assert llm.calls == []            # no network call at all
    assert rec.source == "QA Bank"


def test_heat_pairs_are_second_priority():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm)
    events, rec = collect(engine, "What is the golden rule during captivity?")
    assert [e.kind for e in events] == ["start", "local"]
    assert events[1].source == "HEAT"
    assert "dignity" in events[1].text
    assert llm.calls == []


def test_llm_fallback_receives_compact_grounded_prompt():
    llm = FakeLLM(reply="• Tourniquet first • then airway")
    engine, kb, _ = build_engine(llm)
    events, rec = collect(engine, "How would you treat someone who is bleeding heavily after an attack?")
    kinds = [e.kind for e in events]
    assert kinds[0] == "start" and "ai_start" in kinds and kinds[-1] == "ai_done"
    assert "local" not in kinds
    system, prompt = llm.calls[0]
    assert "Jane Doe" in system                      # CV excerpt in the (short) system prompt
    assert "INTERVIEW QUESTION" in prompt
    assert "tourniquet" in prompt.lower()            # HEAT passage retrieved into the prompt
    assert len(system) < 2000 and len(prompt) < 4000  # compact → fast
    assert rec.source == "Gemini" and "Tourniquet" in rec.answer
    assert engine.history[-1].question.startswith("How would you treat")


def test_medium_confidence_candidate_is_shown_before_ai():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm, {"qa_high_confidence": 1.01, "qa_medium_confidence": 0.3})
    events, _ = collect(engine, "How do you handle a hostile checkpoint?")
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["start", "candidate"]
    assert events[1].matched_question == "How do you handle a hostile checkpoint?"
    assert "ai_done" in kinds
    _, prompt = llm.calls[0]
    assert "Prepared Q&A" in prompt and "Stay calm" in prompt


def test_force_ai_bypasses_local_hit():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm)
    events, _ = collect(engine, "Tell me about yourself.", force_ai=True)
    kinds = [e.kind for e in events]
    assert "local" not in kinds and "ai_done" in kinds
    assert "candidate" in kinds  # the strong local hit is still shown while AI streams


def test_no_llm_falls_back_to_local_suggestions():
    engine, _, _ = build_engine(FakeLLM(available=False))
    events, rec = collect(engine, "How would you treat someone who is bleeding heavily?")
    assert events[-1].kind == "info"
    assert "tourniquet" in events[-1].text.lower() or "Closest matches" in events[-1].text


def test_cancellation_stops_streaming():
    llm = FakeLLM(reply="one two three four five six")
    engine, _, _ = build_engine(llm)
    events = []
    counter = {"n": 0}

    def cancelled():
        counter["n"] += 1
        return counter["n"] > 3

    rec = engine.answer("How would you treat heavy bleeding?", on_event=events.append, is_cancelled=cancelled)
    assert rec is None
    assert "ai_done" not in [e.kind for e in events]


def test_empty_question_reports_error():
    engine, _, _ = build_engine(FakeLLM())
    events, rec = collect(engine, "   ")
    assert events[0].kind == "error" and rec is None


def test_engine_is_thread_safe_under_parallel_questions():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm)
    results = []

    def worker(q):
        results.append(engine.answer(q, on_event=lambda e: None))

    threads = [threading.Thread(target=worker, args=(q,)) for q in [
        "Tell me about yourself.", "What are your salary expectations?", "How do you handle a hostile checkpoint?"] * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 9 and all(r is not None for r in results)


def test_preamble_does_not_dilute_local_match():
    llm = FakeLLM()
    engine, _, _ = build_engine(llm)
    events, rec = collect(engine, "Thanks for joining. So, tell me about yourself?")
    assert [e.kind for e in events] == ["start", "local"]
    assert events[1].confidence >= 0.62 and "eight years" in events[1].text
    assert llm.calls == []


def test_query_variants():
    v = AnswerEngine.query_variants("Great. Thanks for that. Now, how do you handle a hostile checkpoint?")
    assert v[0].startswith("Great.")
    assert "Now, how do you handle a hostile checkpoint?" in v
    assert AnswerEngine.query_variants("Tell me about yourself?") == ["Tell me about yourself?"]


def test_extract_question_drops_filler_lead_in():
    now = 10.0
    det = extract_question([(1, "Thanks for joining us today. Tell me about yourself?", now - 1)], now=now)
    assert det.text == "Tell me about yourself?"
    det = extract_question([(2, "Imagine your convoy is ambushed. What would you do?", now - 1)], now=now)
    assert det.text.startswith("Imagine your convoy")
