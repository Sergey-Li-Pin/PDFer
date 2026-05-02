import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from deep_translator import GoogleTranslator as DeepGoogleTranslator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


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


class GoogleTranslator(BaseTranslator):
    """
    Google Translate wrapper using ``deep-translator`` with local JSON caching.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the GoogleTranslator.

        Args:
            cache_path: Path to the JSON cache file. Defaults to
                        ``output/translation_cache.json`` relative to project root.
            max_retries: Maximum number of retries on network errors.
            retry_delay: Initial delay between retries (doubles each retry).
        """
        self._console = Console()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if cache_path is None:
            root = Path(__file__).resolve().parent.parent
            cache_path = str(root / "output" / "translation_cache.json")
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._load_cache()

        self._translator = DeepGoogleTranslator(source="auto", target="ru")

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(text: str, target_lang: str) -> str:
        """Return a stable hash key for the (text, lang) pair."""
        payload = f"{target_lang}::{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate ``text`` to ``target_lang`` using Google Translate.

        Checks the local cache first, then calls the API with retries.
        """
        if not text or not text.strip():
            return text

        key = self._cache_key(text, target_lang)
        if key in self._cache:
            return self._cache[key]

        # deep-translator target is set at init, but we support dynamic lang
        if self._translator.target != target_lang:
            self._translator = DeepGoogleTranslator(source="auto", target=target_lang)

        last_exception: Exception | None = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                translated = self._translator.translate(text)
                self._cache[key] = translated
                self._save_cache()
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

    def translate_batch(
        self, texts: list[str], target_lang: str
    ) -> dict[str, str]:
        """
        Translate a batch of unique strings efficiently.

        Args:
            texts: List of strings to translate.
            target_lang: Target language code.

        Returns:
            Mapping ``original_text -> translated_text``.
        """
        unique_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
        results: dict[str, str] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self._console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Translating {len(unique_texts)} unique spans...",
                total=len(unique_texts),
            )
            for txt in unique_texts:
                results[txt] = self.translate(txt, target_lang)
                progress.advance(task)

        return results


class OllamaTranslator(BaseTranslator):
    """
    Local LLM translator using the official ``ollama`` Python library.
    Falls back to the original text if Ollama is unreachable.
    """

    def __init__(
        self,
        model: str = "llama3:8b",
        host: str = "http://localhost:11434",
    ):
        """
        Initialize the OllamaTranslator.

        Args:
            model: Model name to use (e.g. ``llama3:8b``, ``mistral``).
            host: Ollama server host URL.
        """
        self._console = Console()
        self.model = model
        self.host = host
        try:
            from ollama import Client
            self._client = Client(host=host)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ollama package is required. Install it: pip install ollama"
            ) from exc

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate ``text`` using the local Ollama LLM.

        Args:
            text: Text to translate.
            target_lang: Target language name or code.

        Returns:
            Translated text, or the original text on failure.
        """
        if not text or not text.strip():
            return text

        prompt = (
            "You are a professional book translator. "
            f"Translate this text to {target_lang}. "
            "Tone: Literary, magical, suitable for children. "
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
            return translated if translated else text
        except Exception as exc:  # noqa: BLE001
            self._console.print(
                f"[yellow]⚠ Ollama request failed ({exc}). "
                f"Falling back to original text.[/yellow]"
            )
            return text

    def translate_batch(
        self, texts: list[str], target_lang: str
    ) -> dict[str, str]:
        """
        Translate a batch of strings via Ollama (one-by-one).

        Args:
            texts: List of strings to translate.
            target_lang: Target language code.

        Returns:
            Mapping ``original_text -> translated_text``.
        """
        unique_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
        results: dict[str, str] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self._console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Translating {len(unique_texts)} paragraphs via Ollama...",
                total=len(unique_texts),
            )
            for txt in unique_texts:
                results[txt] = self.translate(txt, target_lang)
                progress.advance(task)

        return results
