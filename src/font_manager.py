import json
import os
from pathlib import Path


class FontManager:
    """
    Maps original PDF font attributes to local replacement TTF files.

    Simple rule-based logic:
      - If font name contains "Bold"      -> DejaVuSans-Bold.ttf
      - If font name contains "Italic"    -> DejaVuSans-Oblique.ttf
      - If font name contains "Oblique"   -> DejaVuSans-Oblique.ttf
      - Otherwise                         -> DejaVuSans.ttf
    """

    def __init__(self, config_path: str | None = None):
        """
        Load the font mapping configuration.

        Args:
            config_path: Path to ``font_map.json``. Defaults to the one in
                         the project root.
        """
        if config_path is None:
            root = Path(__file__).resolve().parent.parent
            config_path = str(root / "font_map.json")
        self.config_path = config_path
        self._map = self._load_config()

    def _load_config(self) -> dict[str, str]:
        if not os.path.isfile(self.config_path):
            return self._default_map()
        with open(self.config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure all required keys exist
        defaults = self._default_map()
        for key, val in defaults.items():
            data.setdefault(key, val)
        return data

    @staticmethod
    def _default_map() -> dict[str, str]:
        root = Path(__file__).resolve().parent.parent / "assets" / "fonts"
        return {
            "default": str(root / "DejaVuSans.ttf"),
            "regular": str(root / "DejaVuSans.ttf"),
            "bold": str(root / "DejaVuSans-Bold.ttf"),
            "italic": str(root / "DejaVuSans-Oblique.ttf"),
            "bolditalic": str(root / "DejaVuSans-BoldOblique.ttf"),
        }

    def detect_style(self, font_name: str) -> str:
        """
        Detect the style category of a font name.

        Args:
            font_name: Raw font name string from the PDF.

        Returns:
            One of: ``bold``, ``italic``, ``bolditalic``, ``regular``.
        """
        lowered = font_name.lower()
        has_bold = "bold" in lowered
        has_italic = "italic" in lowered
        has_oblique = "oblique" in lowered

        if has_bold and (has_italic or has_oblique):
            return "bolditalic"
        if has_bold:
            return "bold"
        if has_italic or has_oblique:
            return "italic"
        return "regular"

    def get_font_path(self, font_name: str) -> str:
        """
        Return the absolute path to the replacement TTF for the given font name.

        Args:
            font_name: Original font name from the PDF.

        Returns:
            Absolute path to a ``.ttf`` file.
        """
        style = self.detect_style(font_name)
        path = self._map.get(style) or self._map.get("default")
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(self.config_path))
            path = os.path.abspath(os.path.join(base_dir, path))
        return path

    def style_report(self) -> dict[str, str]:
        """Return a mapping of style -> resolved font path for debugging."""
        return {
            style: self.get_font_path(style)
            for style in ("regular", "bold", "italic", "bolditalic")
        }
