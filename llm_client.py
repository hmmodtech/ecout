"""
Thin wrapper around the Gemini SDKs.

* Prefers the current ``google-genai`` SDK, falls back to the legacy
  ``google-generativeai`` package that older installs still have.
* Stateless streaming generation (no ever-growing chat history → constant
  latency) with a *small* system prompt.
* Embeddings for the optional semantic search.

All network errors are swallowed and reported through return values so the
UI never crashes because of the API.
"""
from __future__ import annotations

import os
import re
import threading
import warnings
from typing import Iterator, List, Optional, Sequence

from retrieval import EmbeddingBackend

_NEW_SDK = None
_OLD_SDK = None
_SDK_LOCK = threading.Lock()


def _load_sdks():
    """Import the SDKs lazily (importing them costs ~0.5s)."""
    global _NEW_SDK, _OLD_SDK
    with _SDK_LOCK:
        if _NEW_SDK is None:
            try:
                from google import genai as _g  # type: ignore
                from google.genai import types as _t  # type: ignore

                _NEW_SDK = (_g, _t)
            except Exception:
                _NEW_SDK = False
        if _OLD_SDK is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import google.generativeai as _og  # type: ignore
                _OLD_SDK = _og
            except Exception:
                _OLD_SDK = False
    return _NEW_SDK, _OLD_SDK


def resolve_api_key(config: dict) -> str:
    """API key from config, environment or the legacy ``keys.py`` file."""
    key = (config or {}).get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        try:
            import keys  # type: ignore

            key = getattr(keys, "GEMINI_API_KEY", "") or ""
        except Exception:
            key = ""
    return key.strip()


class GeminiClient:
    """Stateless Gemini text generation + embeddings."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 max_tokens: int = 300, temperature: float = 0.3,
                 embedding_model: str = "gemini-embedding-001", thinking: str = "minimal"):
        self.api_key = (api_key or "").strip()
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.embedding_model = embedding_model
        self.thinking = (thinking or "minimal").lower()
        self._client = None
        self._backend = None  # "new" | "old" | None
        self.last_error: Optional[str] = None
        if self.api_key:
            self._init()

    # -- setup ---------------------------------------------------------------
    def _init(self):
        new_sdk, old_sdk = _load_sdks()
        try:
            if new_sdk:
                genai, _types = new_sdk
                self._client = genai.Client(api_key=self.api_key)
                self._backend = "new"
                return
            if old_sdk:
                old_sdk.configure(api_key=self.api_key)
                self._client = old_sdk
                self._backend = "old"
                return
            self.last_error = "No Gemini SDK installed (pip install google-genai)"
        except Exception as exc:
            self.last_error = f"Gemini init failed: {exc}"
            self._client = None
            self._backend = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def describe(self) -> str:
        if not self.available:
            return "AI: off"
        return f"AI: {self.model_name} ({self._backend} sdk)"

    # -- thinking control ------------------------------------------------------
    def thinking_config(self, types, level: Optional[str] = None):
        """Build a ThinkingConfig that minimises time-to-first-token.

        Gemini 3.x Flash models think at ``medium`` level by default and the
        thinking tokens count against ``max_output_tokens`` – for a
        real-time copilot that means slow (or truncated) answers.
        """
        mode = (level or self.thinking or "minimal").lower()
        if mode in ("default", "auto", ""):
            return None
        name = self.model_name.lower()
        if re.search(r"gemini-2\.[05]", name):
            budget = {"off": 0, "minimal": 0, "low": 256, "medium": 1024, "high": -1}.get(mode, 0)
            return types.ThinkingConfig(thinking_budget=budget)
        if "gemini-3" in name:
            lvl = mode if mode in ("minimal", "low", "medium", "high") else "minimal"
            # 3.7 / 3.8 Flash and 3.x Pro reject "minimal"
            if lvl == "minimal" and (re.search(r"3\.[7-9]", name) or "pro" in name):
                lvl = "low"
            return types.ThinkingConfig(thinking_level=lvl)
        return None

    def effective_max_tokens(self, max_tokens: int) -> int:
        """Thinking tokens are billed against max_output_tokens on 3.x models,
        so keep a safety floor to avoid truncated/empty answers."""
        if "gemini-3" in self.model_name.lower() or "2.5" in self.model_name:
            return max(int(max_tokens), 512)
        return int(max_tokens)

    # -- generation ----------------------------------------------------------
    def stream(self, system_instruction: str, prompt: str,
               max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        """Yield text chunks.  Raises RuntimeError with a readable message."""
        if not self.available:
            raise RuntimeError(self.last_error or "Gemini API key is not configured")
        max_tokens = self.effective_max_tokens(max_tokens or self.max_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        if self._backend == "new":
            _genai, types = _load_sdks()[0]
            attempts = [self.thinking_config(types)]
            if attempts[0] is not None:
                attempts.append(self.thinking_config(types, "low"))
                attempts.append(None)
            for i, thinking in enumerate(attempts):
                cfg = types.GenerateContentConfig(
                    system_instruction=system_instruction or None,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=thinking,
                )
                yielded = False
                try:
                    stream = self._client.models.generate_content_stream(
                        model=self.model_name, contents=prompt, config=cfg)
                    for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            yielded = True
                            yield text
                    return
                except Exception as exc:
                    # Retry only when the model rejected our thinking settings
                    # (e.g. "minimal" unsupported) and nothing was streamed yet.
                    retryable = (not yielded and "thinking" in str(exc).lower()
                                 and i + 1 < len(attempts))
                    if not retryable:
                        raise
        else:
            model = self._client.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction or None,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            for chunk in model.generate_content(prompt, stream=True):
                try:
                    text = chunk.text
                except Exception:
                    text = None
                if text:
                    yield text

    def generate(self, system_instruction: str, prompt: str, **kw) -> str:
        return "".join(self.stream(system_instruction, prompt, **kw))

    # -- embeddings ----------------------------------------------------------
    def embed(self, texts: Sequence[str], kind: str = "document") -> Optional[List[List[float]]]:
        if not self.available or not texts:
            return None
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        try:
            if self._backend == "new":
                _genai, types = _load_sdks()[0]
                resp = self._client.models.embed_content(
                    model=self.embedding_model,
                    contents=list(texts),
                    config=types.EmbedContentConfig(task_type=task),
                )
                return [list(e.values) for e in resp.embeddings]
            model = self.embedding_model
            if not model.startswith("models/"):
                model = "models/" + model
            resp = self._client.embed_content(model=model, content=list(texts), task_type=task)
            emb = resp["embedding"]
            if emb and isinstance(emb[0], (int, float)):
                return [list(emb)]
            return [list(e) for e in emb]
        except Exception as exc:
            self.last_error = f"embedding failed: {exc}"
            print(f"[Gemini] {self.last_error}")
            return None


class GeminiEmbeddings(EmbeddingBackend):
    """Adapter so :class:`retrieval.SearchIndex` can use Gemini embeddings."""

    def __init__(self, client: GeminiClient):
        self.client = client
        self.name = f"gemini:{client.embedding_model}"

    def embed(self, texts: Sequence[str], kind: str = "document") -> Optional[List[List[float]]]:
        return self.client.embed(texts, kind)
