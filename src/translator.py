import os

for var in ["all_proxy", "ALL_PROXY", "https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"]:
    os.environ.pop(var, None)

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from deep_translator import GoogleTranslator as DeepGoogleTranslator
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()


class TranslationCache:
    """Thread-safe JSON cache for translated strings."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        else:
            self._cache = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _key(text: str, target_lang: str) -> str:
        payload = f"{target_lang}::{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, text: str, target_lang: str) -> str | None:
        with self._lock:
            return self._cache.get(self._key(text, target_lang))

    def set(self, text: str, target_lang: str, translated: str) -> None:
        with self._lock:
            self._cache[self._key(text, target_lang)] = translated
            self._save()


class BaseTranslator(ABC):
    """Abstract base class for translation engines."""

    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate the given text to the target language.

        Args:
            text: The text to translate.
            target_lang: Target language code (e.g. 'ru', 'en').

        Returns:
            Translated text.
        """
        ...

    def translate_batch(
        self, texts: list[str], target_lang: str, threads: int = 4
    ) -> dict[str, str]:
        """
        Translate a batch of strings in parallel.

        Args:
            texts: List of strings to translate.
            target_lang: Target language code.
            threads: Number of parallel workers.

        Returns:
            Mapping ``original_text -> translated_text``.
        """
        unique_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
        results: dict[str, str] = {}
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Parallel translation in progress...",
                total=len(unique_texts),
            )
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(self.translate, txt, target_lang): txt
                    for txt in unique_texts
                }
                for future in as_completed(futures):
                    txt = futures[future]
                    try:
                        results[txt] = future.result()
                    except Exception:  # noqa: BLE001
                        results[txt] = txt
                    progress.advance(task)

        return results


class GoogleTranslator(BaseTranslator):
    """
    Google Translate wrapper using ``deep-translator`` with thread-safe JSON caching.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if cache_path is None:
            root = Path(__file__).resolve().parent.parent
            cache_path = str(root / "output" / "translation_cache.json")
        self._cache = TranslationCache(cache_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._translator = DeepGoogleTranslator(source="auto", target="ru")
        self._console = Console()

    def translate(self, text: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text

        cached = self._cache.get(text, target_lang)
        if cached is not None:
            return cached

        if self._translator.target != target_lang:
            self._translator = DeepGoogleTranslator(source="auto", target=target_lang)

        last_exception: Exception | None = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                translated = self._translator.translate(text)
                self._cache.set(text, target_lang, translated)
                return translated
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                continue

        self._console.print(
            f"[yellow]⚠ Translation failed after {self.max_retries} attempts. "
            f"Falling back to original text.[/yellow]"
        )
        return text


class OllamaTranslator(BaseTranslator):
    """
    Local LLM translator using the official ``ollama`` Python library.
    Adds a small intra-thread delay to reduce CPU thrashing.
    Uses thread-safe JSON caching and disables system proxy trust.
    """

    def __init__(
        self,
        model: str = "llama3:8b",
        host: str = "http://localhost:11434",
        intra_delay: float = 0.5,
        cache_path: str | None = None,
    ):
        self.model = model
        self.host = host
        self.intra_delay = intra_delay
        self._console = Console()
        if cache_path is None:
            root = Path(__file__).resolve().parent.parent
            cache_path = str(root / "output" / "translation_cache.json")
        self._cache = TranslationCache(cache_path)
        try:
            from ollama import Client
            self._client = Client(host=host, trust_env=False)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ollama package is required. Install it: pip install ollama"
            ) from exc

    def translate(
        self, text: str, target_lang: str, draft_text: str | None = None
    ) -> str:
        if not text or not text.strip():
            return text

        cached = self._cache.get(text, target_lang)
        if cached is not None:
            return cached

        # Small stagger to prevent CPU thrashing when many threads hit Ollama
        time.sleep(self.intra_delay)

        if draft_text:
            prompt = (
                f"I have a draft translation: '{draft_text}'. "
                f"The original text was: '{text}'. "
                f"Rewrite this into high-quality, natural {target_lang}. "
                "IMPORTANT: The result must be roughly the same length as the original "
                "to fit in a PDF box. "
                "Return ONLY the polished translation."
            )
        else:
            prompt = (
                "You are a professional universal translator. "
                f"Translate the following text to {target_lang}. "
                "Preserve formatting, tone, and nuances. "
                "Return ONLY the translation."
                f"\n\n{text}"
            )

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            translated = response.get("response", "").strip()
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            if translated:
                self._cache.set(text, target_lang, translated)
                return translated
            return text
        except Exception as exc:  # noqa: BLE001
            self._console.print(
                f"[yellow]⚠ Ollama request failed ({exc}). "
                f"Falling back to original text.[/yellow]"
            )
            return text


class OpenRouterTranslator(BaseTranslator):
    """
    Cloud LLM translator using the OpenRouter API.
    Uses thread-safe JSON caching and clean HTTPS requests (no proxy).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._console = Console()
        if cache_path is None:
            root = Path(__file__).resolve().parent.parent
            cache_path = str(root / "output" / "translation_cache.json")
        self._cache = TranslationCache(cache_path)
        if not self.api_key:
            self._console.print(
                "[yellow]⚠ OPENROUTER_API_KEY not found. "
                "Set it in .env or pass it directly.[/yellow]"
            )

    def translate(
        self, text: str, target_lang: str, draft_text: str | None = None
    ) -> str:
        if not text or not text.strip():
            return text

        cached = self._cache.get(text, target_lang)
        if cached is not None:
            return cached

        if draft_text:
            prompt = (
                f"You are a professional translator. I have a draft: '{draft_text}'. "
                f"The original: '{text}'. "
                f"Rewrite it into natural, high-quality {target_lang}. "
                "Keep it roughly the same length. "
                "Return ONLY the translation."
            )
        else:
            prompt = (
                "You are a professional universal translator. "
                f"Translate the following text to {target_lang}. "
                "Preserve formatting, tone, and nuances. "
                "Return ONLY the translation."
                f"\n\n{text}"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://pdf-master-translate.local",
            "X-Title": "PDF Master Translate",
        }

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                translated = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if translated.startswith('"') and translated.endswith('"'):
                    translated = translated[1:-1]
                if translated:
                    self._cache.set(text, target_lang, translated)
                    return translated
                return text
        except Exception as exc:  # noqa: BLE001
            self._console.print(
                f"[yellow]⚠ OpenRouter request failed ({exc}). "
                f"Falling back to original text.[/yellow]"
            )
            return text
