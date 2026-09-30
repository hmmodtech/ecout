"""
Lexical + (optional) semantic retrieval over the knowledge base.

The goal is *instant* local answers: matching a spoken interview question
against the Q&A bank must take a few milliseconds, with no network round
trip.  We therefore use:

* a light normaliser (lower-case, punctuation removal, stop-words, a tiny
  suffix stemmer and an interview/HEAT synonym map),
* BM25 for candidate generation,
* an IDF-weighted coverage / fuzzy re-ranker that yields a calibrated
  confidence in ``[0, 1]`` so the caller can decide between
  "answer straight from the file" and "ask the LLM".

An optional embedding backend (Gemini embeddings) can be plugged in to catch
paraphrases that share no vocabulary.  It is strictly additive – when it is
missing or fails, everything still works lexically.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from knowledge_base import Passage, QAPair

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
STOPWORDS = set("""
a an the and or but if then so of to in on at for from by with about as into
like through after over between out against during without before under around
among is are was were be been being am do does did doing have has had having
i me my mine myself we us our ours you your yours he him his she her hers it its
they them their theirs this that these those there here what which who whom whose
when where how would could should can may might must shall will just very
really also too any some all each every both few more most other such no nor not
only own same than s t don ve ll re d m please well okay ok um uh yeah yes right
tell me us describe explain walk give share talk elaborate say know think mean
kind sort bit little lot thing things something anything
هل ما ماذا لماذا كيف متى أين اين من في على عن إلى الى مع هذا هذه ذلك تلك أن ان
كان كانت هو هي هم أنا انا نحن أنت انت لو إذا اذا ثم أو او و يا
""".split())

# Words that mark a question as *behavioural* – they add no meaning for matching
_FILLER_PHRASES = [
    r"\btell (?:me|us) (?:a bit |a little |more )?about\b",
    r"\bcan you (?:please )?(?:tell|describe|explain|walk|give|share)\b",
    r"\bcould you (?:please )?(?:tell|describe|explain|walk|give|share)\b",
    r"\bwould you (?:please )?(?:tell|describe|explain|walk|give|share)\b",
    r"\bgive (?:me|us) an example of\b",
    r"\bdescribe a (?:time|situation) (?:when|where|in which)\b",
    r"\btell (?:me|us) about a (?:time|situation) (?:when|where|in which)\b",
    r"\bwalk (?:me|us) through\b",
    r"\bin your (?:own )?(?:words|opinion|experience|view)\b",
    r"\bi(?:'d| would) like to know\b",
    r"\bi was wondering\b",
    r"\bso\b,?\s*",
    r"\bokay\b,?\s*",
    r"\balright\b,?\s*",
]
_FILLER_RE = re.compile("|".join(_FILLER_PHRASES), re.IGNORECASE)

# Canonical synonym groups (interview + hostile-environment training domain)
_SYNONYM_GROUPS: List[List[str]] = [
    ["handle", "deal", "manage", "cope", "respond", "react", "approach", "tackle", "address"],
    ["hostile", "aggressive", "threatening", "violent", "angry", "confrontational"],
    ["example", "instance", "situation", "scenario", "case", "experience", "story"],
    ["yourself", "background", "introduce", "introduction", "profile", "bio"],
    ["weakness", "weaknesses", "flaw", "improve", "improvement", "development", "shortcoming"],
    ["strength", "strengths", "strong", "good", "best", "skill", "skills"],
    ["conflict", "disagreement", "argument", "dispute", "clash", "tension"],
    ["team", "teammate", "colleague", "coworker", "peer", "group"],
    ["pressure", "stress", "stressful", "deadline", "tight"],
    ["checkpoint", "roadblock", "vcp", "barrier", "control point"],
    ["kidnap", "kidnapping", "abduction", "abducted", "hostage", "captivity", "captive", "detained", "detention"],
    ["medical", "casualty", "injury", "injured", "wounded", "bleeding", "first aid", "trauma", "tourniquet"],
    ["mine", "mines", "ied", "uxo", "explosive", "explosives", "ordnance", "bomb", "landmine", "minefield"],
    ["evacuate", "evacuation", "extract", "extraction", "withdraw", "leave", "exit"],
    ["ambush", "attack", "assault", "fire", "shooting", "gunfire", "shot", "crossfire", "contact"],
    ["convoy", "vehicle", "vehicles", "car", "movement", "journey", "travel", "route", "road"],
    ["risk", "threat", "danger", "hazard", "security"],
    ["plan", "planning", "prepare", "preparation", "assessment", "assess"],
    ["communication", "communicate", "radio", "comms", "check-in", "checkin", "report"],
    ["leader", "leadership", "lead", "manager", "supervisor", "command"],
    ["fail", "failure", "mistake", "error", "wrong", "lesson", "learned"],
    ["motivate", "motivation", "reason", "interest", "interested", "apply", "applying", "join", "joining",
     "organisation", "organization", "company", "employer", "ngo", "agency"],
    ["salary", "pay", "compensation", "expectations", "package"],
    ["goal", "goals", "future", "years", "ambition", "career"],
    ["crowd", "mob", "riot", "demonstration", "protest", "civil unrest", "unrest"],
    ["negotiate", "negotiation", "de-escalate", "deescalate", "de-escalation", "calm"],
    ["culture", "cultural", "local", "customs", "community", "population"],
    ["woman", "women", "female", "gender"],
    ["child", "children", "kid", "kids", "minor"],
    ["weapon", "weapons", "gun", "guns", "armed", "rifle", "knife"],
    ["natural", "earthquake", "flood", "disaster", "emergency"],
    ["sexual", "harassment", "assault", "abuse", "exploitation"],
    ["prioritize", "prioritise", "priority", "priorities", "urgent", "important"],
]
_SYNONYMS: Dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    _canon = _group[0]
    for _w in _group:
        _SYNONYMS.setdefault(_w, _canon)

_TOKEN_RE = re.compile(r"[a-z0-9\u0600-\u06FF]+(?:['\-][a-z0-9]+)*")


def _stem(word: str) -> str:
    """Very small English suffix stripper (not Porter, but predictable)."""
    if len(word) <= 3 or not word.isascii():
        return word
    for suf, repl in (("ations", "ate"), ("ation", "ate"), ("ising", "ise"), ("izing", "ize"),
                      ("ingly", ""), ("edly", ""), ("ments", "ment"), ("ness", ""),
                      ("ities", "ity"), ("ies", "y"), ("ing", ""), ("ed", ""), ("ers", "er"),
                      ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)] + repl
            break
    if word.endswith("i") and len(word) > 4:  # "tri" -> "try"
        word = word[:-1] + "y"
    return word


def canonical_tokens(text: str, drop_stopwords: bool = True) -> List[str]:
    """Tokenise + normalise a piece of text into canonical content tokens."""
    if not text:
        return []
    low = text.lower().replace("’", "'")
    low = _FILLER_RE.sub(" ", low)
    tokens: List[str] = []
    for tok in _TOKEN_RE.findall(low):
        tok = tok.strip("'-")
        if not tok:
            continue
        if drop_stopwords and tok in STOPWORDS:
            continue
        canon = _SYNONYMS.get(tok)
        if canon is None:
            stem = _stem(tok)
            canon = _SYNONYMS.get(stem, stem)
        tokens.append(canon)
    return tokens


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for d in self.docs:
            counts: Dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf: Dict[str, float] = {
            t: math.log(1.0 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }
        self.max_idf = max(self.idf.values()) if self.idf else 1.0

    def idf_of(self, token: str) -> float:
        # unseen tokens are rare → high idf (helps coverage weighting)
        return self.idf.get(token, self.max_idf)

    def score(self, query: Sequence[str], idx: int) -> float:
        if not self.n:
            return 0.0
        counts = self.tf[idx]
        dl = len(self.docs[idx])
        s = 0.0
        for t in query:
            f = counts.get(t)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def search(self, query: Sequence[str], top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.n or not query:
            return []
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #
def _weighted_coverage(query: Sequence[str], doc: Sequence[str], idf: Callable[[str], float]) -> float:
    """Share of the query's (idf-weighted) tokens present in ``doc``."""
    if not query:
        return 0.0
    doc_set = set(doc)
    num = sum(idf(t) for t in set(query) if t in doc_set)
    den = sum(idf(t) for t in set(query))
    return num / den if den else 0.0


def _token_sort_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = " ".join(sorted(a)), " ".join(sorted(b))
    return SequenceMatcher(None, sa, sb).ratio()


def _sequence_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, list(a), list(b)).ratio()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class QAMatch:
    pair: QAPair
    confidence: float
    lexical: float = 0.0
    semantic: float = 0.0
    bm25: float = 0.0


@dataclass
class PassageMatch:
    passage: Passage
    score: float


@dataclass
class SearchResult:
    query: str
    qa_matches: List[QAMatch] = field(default_factory=list)
    passages: List[PassageMatch] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def best(self) -> Optional[QAMatch]:
        return self.qa_matches[0] if self.qa_matches else None


# --------------------------------------------------------------------------- #
# Optional embedding backend
# --------------------------------------------------------------------------- #
class EmbeddingBackend:
    """Interface for semantic search backends.

    ``embed(texts, kind)`` returns one vector per text (``kind`` is
    ``"document"`` or ``"query"``) or ``None`` on failure.
    """

    name = "none"

    def embed(self, texts: Sequence[str], kind: str = "document") -> Optional[List[List[float]]]:  # pragma: no cover
        raise NotImplementedError


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def _semantic_confidence(cos: float) -> float:
    """Map cosine similarity of embedding models to a [0,1] confidence.

    Paraphrases of the same question sit around 0.85-0.95, unrelated
    questions in the same domain around 0.5-0.65."""
    return max(0.0, min(1.0, (cos - 0.60) / (0.92 - 0.60)))


# --------------------------------------------------------------------------- #
# Search index
# --------------------------------------------------------------------------- #
class SearchIndex:
    """Index over Q&A pairs and passages with calibrated confidence scores."""

    def __init__(self, qa_pairs: Sequence[QAPair], passages: Sequence[Passage],
                 embedding_backend: Optional[EmbeddingBackend] = None,
                 cache_dir: Optional[str] = None):
        self.qa_pairs = list(qa_pairs)
        self.passages = list(passages)
        self._qa_tokens = [canonical_tokens(p.question) for p in self.qa_pairs]
        # Answers contribute to BM25 recall (a question may be phrased using
        # words that only occur in the stored answer), but with low weight.
        self._qa_docs = [q + canonical_tokens(p.answer)[:40] for q, p in zip(self._qa_tokens, self.qa_pairs)]
        self._qa_bm25 = BM25(self._qa_docs) if self._qa_docs else None
        self._passage_tokens = [canonical_tokens(f"{p.title} {p.text}") for p in self.passages]
        self._passage_bm25 = BM25(self._passage_tokens) if self._passage_tokens else None

        self._embedding_backend = embedding_backend
        self._qa_vectors: Optional[List[List[float]]] = None
        self._embed_lock = threading.Lock()
        self._cache_dir = cache_dir
        self.semantic_ready = False

    # -- semantic index (background) ---------------------------------------
    def _cache_path(self) -> Optional[str]:
        if not self._cache_dir or not self._embedding_backend:
            return None
        digest = hashlib.sha1(
            (self._embedding_backend.name + "\n" + "\n".join(p.question for p in self.qa_pairs)).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(self._cache_dir, f".qa_embeddings_{digest}.json")

    def build_semantic_index(self) -> bool:
        """Embed all Q&A questions (cached on disk).  Safe to call in a thread."""
        if not self._embedding_backend or not self.qa_pairs:
            return False
        with self._embed_lock:
            if self._qa_vectors is not None:
                return True
            path = self._cache_path()
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        vectors = json.load(f)
                    if len(vectors) == len(self.qa_pairs):
                        self._qa_vectors = vectors
                        self.semantic_ready = True
                        return True
                except Exception:
                    pass
            try:
                vectors: List[List[float]] = []
                batch = 64
                questions = [p.question for p in self.qa_pairs]
                for i in range(0, len(questions), batch):
                    part = self._embedding_backend.embed(questions[i:i + batch], "document")
                    if not part:
                        return False
                    vectors.extend(part)
                self._qa_vectors = vectors
                self.semantic_ready = True
                if path:
                    try:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(vectors, f)
                    except Exception:
                        pass
                return True
            except Exception as exc:
                print(f"[SearchIndex] semantic index failed: {exc}")
                return False

    def semantic_scores(self, query: str) -> Optional[Dict[int, float]]:
        if not self.semantic_ready or not self._qa_vectors or not self._embedding_backend:
            return None
        try:
            vec = self._embedding_backend.embed([query], "query")
        except Exception:
            return None
        if not vec:
            return None
        q = vec[0]
        return {i: _cosine(q, v) for i, v in enumerate(self._qa_vectors)}

    # -- lexical search ------------------------------------------------------
    def _lexical_qa(self, query_tokens: List[str], top_k: int, categories: Optional[Iterable[str]]) -> List[QAMatch]:
        if not self._qa_bm25 or not query_tokens:
            return []
        cats = set(categories) if categories else None
        candidates = self._qa_bm25.search(query_tokens, top_k=max(top_k * 4, 20))
        # Always consider fuzzy candidates too (BM25 misses zero-overlap stems)
        cand_idx = {i for i, _ in candidates}
        if len(cand_idx) < top_k:
            cand_idx.update(range(len(self.qa_pairs)))
        idf = self._qa_bm25.idf_of
        top_bm25 = candidates[0][1] if candidates else 0.0
        results: List[QAMatch] = []
        for i in cand_idx:
            pair = self.qa_pairs[i]
            if cats and pair.category not in cats:
                continue
            doc_q = self._qa_tokens[i]
            cov = _weighted_coverage(query_tokens, doc_q, idf)
            rev = _weighted_coverage(doc_q, query_tokens, idf)
            sort_ratio = _token_sort_ratio(query_tokens, doc_q)
            seq_ratio = _sequence_ratio(query_tokens, doc_q)
            bm = self._qa_bm25.score(query_tokens, i)
            bm_norm = bm / top_bm25 if top_bm25 else 0.0
            lexical = 0.45 * cov + 0.20 * rev + 0.15 * sort_ratio + 0.10 * seq_ratio + 0.10 * bm_norm
            # A single shared rare word must not look like a match.
            if len(set(query_tokens) & set(doc_q)) <= 1 and len(query_tokens) >= 3:
                lexical *= 0.6
            results.append(QAMatch(pair, confidence=lexical, lexical=lexical, bm25=bm))
        results.sort(key=lambda m: m.confidence, reverse=True)
        return results[:top_k]

    def _lexical_passages(self, query_tokens: List[str], top_k: int) -> List[PassageMatch]:
        if not self._passage_bm25 or not query_tokens:
            return []
        hits = self._passage_bm25.search(query_tokens, top_k=top_k * 2)
        if not hits:
            return []
        top = hits[0][1] or 1.0
        idf = self._passage_bm25.idf_of
        out: List[PassageMatch] = []
        for i, s in hits:
            cov = _weighted_coverage(query_tokens, self._passage_tokens[i], idf)
            out.append(PassageMatch(self.passages[i], 0.6 * (s / top) + 0.4 * cov))
        out.sort(key=lambda m: m.score, reverse=True)
        return out[:top_k]

    def search(self, query: str, top_k: int = 5, categories: Optional[Iterable[str]] = None,
               use_semantic: bool = True, passage_k: int = 4) -> SearchResult:
        t0 = time.perf_counter()
        q_tokens = canonical_tokens(query)
        qa = self._lexical_qa(q_tokens, top_k, categories)
        passages = self._lexical_passages(q_tokens, passage_k)
        if use_semantic and self.semantic_ready:
            sem = self.semantic_scores(query)
            if sem:
                by_idx = {id(m.pair): m for m in qa}
                for i, cos in sem.items():
                    pair = self.qa_pairs[i]
                    if categories and pair.category not in set(categories):
                        continue
                    conf = _semantic_confidence(cos)
                    m = by_idx.get(id(pair))
                    if m is None:
                        if conf >= 0.35:
                            m = QAMatch(pair, confidence=0.0)
                            qa.append(m)
                            by_idx[id(pair)] = m
                        else:
                            continue
                    m.semantic = conf
                    # Agreeing signals reinforce each other; otherwise trust the stronger one.
                    m.confidence = max(m.lexical, conf, 0.5 * (m.lexical + conf) + 0.15 * min(m.lexical, conf))
                qa.sort(key=lambda m: m.confidence, reverse=True)
                qa = qa[:top_k]
        result = SearchResult(query=query, qa_matches=qa, passages=passages)
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result
