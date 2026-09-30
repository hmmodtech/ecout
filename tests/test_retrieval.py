import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base import QAPair, chunk_passages  # noqa: E402
from retrieval import EmbeddingBackend, SearchIndex, canonical_tokens  # noqa: E402

QA = [
    QAPair("Tell me about yourself.", "I am a security professional with eight years in hostile environments.", "qa", "QA"),
    QAPair("How do you handle a hostile checkpoint?", "Stay calm, hands visible, comply, report to ops.", "qa", "QA"),
    QAPair("What is your biggest weakness?", "I used to take on too much; now I delegate.", "qa", "QA"),
    QAPair("Describe a time you dealt with an aggressive crowd.", "I withdrew the team and de-escalated via elders.", "qa", "QA"),
    QAPair("What would you do if your convoy was ambushed?", "Drive through if possible, otherwise take cover and report.", "qa", "QA"),
    QAPair("Why do you want to work with our organisation?", "Your mandate matches my humanitarian security experience.", "qa", "QA"),
    QAPair("What are your salary expectations?", "I am flexible and focused on the role.", "qa", "QA"),
    QAPair("How do you prioritise competing tasks?", "Safety first, then time-critical deliverables.", "qa", "QA"),
    QAPair("What is the first thing you do at a mine incident?", "Stop, do not move, warn others.", "heat", "HEAT"),
    QAPair("How often do you do radio check-ins on the move?", "Every 30 minutes.", "heat", "HEAT"),
]

HEAT_TEXT = """# Kidnapping

The first 24 hours of a kidnapping are the most dangerous. Stay calm, do not resist physically, and
try to humanise yourself to the captors. Follow the proof-of-life protocol agreed with your organisation.

# Medical

Catastrophic bleeding is treated first: apply a tourniquet high and tight, note the time.
Then airway, breathing, circulation. Call for evacuation on the emergency channel.

# Checkpoints

Approach slowly, stop where indicated and keep both hands visible. Never argue with armed personnel.
"""


def make_index(backend=None):
    return SearchIndex(QA, chunk_passages(HEAT_TEXT, "heat", "HEAT", max_chars=300), embedding_backend=backend)


def test_canonical_tokens_normalises_fillers_and_synonyms():
    toks = canonical_tokens("So, tell me about a time when you had to deal with an aggressive crowd?")
    assert "handle" in toks          # deal -> handle
    assert "hostile" in toks         # aggressive -> hostile
    assert "crowd" in toks
    assert "tell" not in toks and "about" not in toks


def test_exact_question_scores_high():
    idx = make_index()
    res = idx.search("How do you handle a hostile checkpoint?", categories=["QA"], use_semantic=False)
    assert res.best.pair.question == "How do you handle a hostile checkpoint?"
    assert res.best.confidence >= 0.9
    assert res.elapsed_ms < 50


def test_paraphrase_matches_right_pair():
    idx = make_index()
    cases = {
        "Okay so, what's your approach when you deal with an aggressive checkpoint?": "How do you handle a hostile checkpoint?",
        "Can you tell us a little bit about yourself?": "Tell me about yourself.",
        "What would you say is your greatest weakness?": "What is your biggest weakness?",
        "Give me an example of a situation where you managed an angry mob.": "Describe a time you dealt with an aggressive crowd.",
        "Imagine your vehicles come under attack on the road, what do you do?": "What would you do if your convoy was ambushed?",
        "Why are you interested in joining us?": "Why do you want to work with our organisation?",
    }
    for query, expected in cases.items():
        res = idx.search(query, categories=["QA"], use_semantic=False)
        assert res.best is not None, query
        assert res.best.pair.question == expected, f"{query!r} → {res.best.pair.question!r} ({res.best.confidence:.2f})"
        assert res.best.confidence >= 0.4, f"{query!r} confidence {res.best.confidence:.2f}"
    # Hard paraphrases with almost no shared vocabulary: still ranked first,
    # but deliberately *not* confident (that is what the semantic backend / AI fallback are for).
    res = idx.search("How do you decide what to do first when everything is urgent?", categories=["QA"], use_semantic=False)
    assert res.best.pair.question == "How do you prioritise competing tasks?"
    assert res.best.confidence < 0.62


def test_unrelated_question_scores_low():
    idx = make_index()
    res = idx.search("Which programming languages do you know for backend development?", categories=["QA"], use_semantic=False)
    assert res.best is None or res.best.confidence < 0.4


def test_category_filter_and_heat_pairs():
    idx = make_index()
    res = idx.search("What do you do first at a mine incident?", categories=["QA"], use_semantic=False)
    assert res.best is None or res.best.pair.category == "QA"
    res = idx.search("What do you do first at a mine incident?", categories=["HEAT"], use_semantic=False)
    assert res.best.pair.question.startswith("What is the first thing you do at a mine")
    assert res.best.confidence >= 0.6


def test_passage_retrieval_from_heat_material():
    idx = make_index()
    res = idx.search("How do you treat catastrophic bleeding?", use_semantic=False)
    assert res.passages
    assert res.passages[0].passage.title == "Medical"
    res = idx.search("What should I do in the first hours if I am kidnapped?", use_semantic=False)
    assert res.passages[0].passage.title == "Kidnapping"


class FakeEmbeddings(EmbeddingBackend):
    """Deterministic 'semantic' backend: embeds by a hand made topic map."""

    name = "fake"
    topics = {
        "yourself": 0, "background": 0, "introduce": 0,
        "checkpoint": 1, "hostile": 1,
        "weakness": 2, "flaw": 2, "improve": 2,
        "crowd": 3, "mob": 3,
        "convoy": 4, "ambush": 4,
        "organisation": 5, "join": 5,
        "salary": 6, "money": 6, "compensation": 6,
        "prioritise": 7, "urgent": 7,
        "mine": 8, "radio": 9,
    }

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind="document"):
        self.calls += 1
        out = []
        for t in texts:
            vec = [0.05] * 10
            for w, i in self.topics.items():
                if w in t.lower():
                    vec[i] += 1.0
            out.append(vec)
        return out


def test_semantic_backend_upgrades_zero_overlap_paraphrase():
    backend = FakeEmbeddings()
    idx = make_index(backend)
    assert idx.build_semantic_index()
    assert idx.semantic_ready
    # "money" shares no token with "salary expectations" lexically
    res = idx.search("How much money are you looking for?", categories=["QA"], use_semantic=True)
    assert res.best.pair.question == "What are your salary expectations?"
    assert res.best.semantic > 0.9
    assert res.best.confidence >= 0.62


def test_semantic_index_cache(tmp_path):
    backend = FakeEmbeddings()
    idx = SearchIndex(QA, [], embedding_backend=backend, cache_dir=str(tmp_path))
    assert idx.build_semantic_index()
    calls = backend.calls
    idx2 = SearchIndex(QA, [], embedding_backend=backend, cache_dir=str(tmp_path))
    assert idx2.build_semantic_index()
    assert backend.calls == calls  # loaded from disk


def test_empty_index_is_safe():
    idx = SearchIndex([], [])
    res = idx.search("anything?")
    assert res.best is None and res.passages == []
