import argparse
import os
import sys

import fitz
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from translator import BaseTranslator, GoogleTranslator

DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "DejaVuSans.ttf"
)


class PDFProcessor:
    """
    Extract text blocks with coordinates and styles from PDF documents,
    generate visual debug output, process translations, and fit translated
    text to original bounding boxes via dynamic font scaling.
    """

    def __init__(self, file_path: str, font_path: str | None = None):
        """
        Initialize the PDFProcessor and open the PDF document.

        Args:
            file_path: Path to the PDF file to process.
            font_path: Path to an external TTF font. Defaults to
                       ``assets/fonts/DejaVuSans.ttf``.
        """
        self.file_path = file_path
        self.font_path = font_path or DEFAULT_FONT_PATH
        self.console = Console()
        self._doc: fitz.Document | None = None
        self._font: fitz.Font | None = None

    def _open(self) -> fitz.Document:
        """Lazy-open the PDF document."""
        if self._doc is None:
            self._doc = fitz.open(self.file_path)
        return self._doc

    def _get_font(self) -> fitz.Font:
        """Lazy-load the external TTF font for width calculations."""
        if self._font is None:
            self._font = fitz.Font(fontfile=self.font_path)
        return self._font

    def extract_text(self) -> str:
        """
        Extract text from the PDF.

        Returns:
            Extracted text as a single string.
        """
        doc = self._open()
        return "\n".join(page.get_text() for page in doc)

    def extract_pages(self) -> list[str]:
        """
        Extract text page by page.

        Returns:
            A list of strings, one per page.
        """
        doc = self._open()
        return [page.get_text() for page in doc]

    def extract_layout(self) -> list[dict]:
        """
        Extract text spans with their bounding boxes and style info.

        Returns:
            A list of dictionaries, each containing:
            - page: page number (0-based)
            - text: the span text
            - bbox: bounding box as a tuple (x0, y0, x1, y1)
            - font_size: font size
            - font_name: font name
        """
        doc = self._open()
        spans = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Extracting layout from {os.path.basename(self.file_path)}...",
                total=len(doc),
            )
            for page_num, page in enumerate(doc):
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            spans.append(
                                {
                                    "page": page_num,
                                    "text": span.get("text", ""),
                                    "bbox": tuple(span["bbox"]),
                                    "font_size": span.get("size", 0.0),
                                    "font_name": span.get("font", ""),
                                }
                            )
                progress.advance(task)

        return spans

    def calculate_optimal_font_size(
        self, text: str, target_width: float, original_font_size: float
    ) -> float:
        """
        Dynamically reduce font size until ``text`` fits inside ``target_width``.

        Args:
            text: The (translated) text to measure.
            target_width: Available width in points (bbox width).
            original_font_size: Starting font size.

        Returns:
            The largest font size ≤ ``original_font_size`` that fits,
            or ``6.0`` as a hard floor.
        """
        font = self._get_font()
        font_size = original_font_size
        MIN_SIZE = 6.0

        while font_size > MIN_SIZE:
            measured = font.text_length(text, fontsize=font_size)
            if measured <= target_width:
                break
            font_size -= 0.5

        return max(font_size, MIN_SIZE)

    def visualize_layout(
        self, output_path: str | None = None, layout: list[dict] | None = None
    ) -> str:
        """
        Create a copy of the PDF with red rectangles around every text span.
        If ``layout`` contains translated text and ``final_font_size``,
        ghost-prints the translated string inside the box in blue.

        Args:
            output_path: Destination file path. Defaults to ``output/debug_layout.pdf``.
            layout: Pre-computed layout list (e.g. from ``process_translation``).
                    If ``None``, ``extract_layout()`` is called.

        Returns:
            The path to the generated debug PDF.
        """
        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "output"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "debug_layout.pdf")

        if layout is None:
            layout = self.extract_layout()

        debug_doc = fitz.open(self.file_path)  # work on a fresh copy

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Rendering debug layout...", total=len(layout)
            )
            for span in layout:
                page = debug_doc[span["page"]]
                rect = fitz.Rect(span["bbox"])
                page.draw_rect(rect, color=(1, 0, 0), width=1.5)

                if "translated_text" in span and "final_font_size" in span:
                    origin = span.get("origin", (span["bbox"][0], span["bbox"][3]))
                    page.insert_text(
                        origin,
                        span["translated_text"],
                        fontsize=span["final_font_size"],
                        color=(0, 0, 1),
                        fontfile=self.font_path,
                    )
                progress.advance(task)

        debug_doc.save(output_path)
        debug_doc.close()
        return output_path

    def process_translation(
        self, translator: BaseTranslator, target_lang: str
    ) -> list[dict]:
        """
        Translate each unique text span, compute ``final_font_size`` so the
        translated string fits the original bbox, and attach the results.

        Args:
            translator: An instance of a ``BaseTranslator`` subclass.
            target_lang: Target language code (e.g. 'ru', 'en').

        Returns:
            The layout list with added fields:
            ``translated_text``, ``final_font_size``, and ``origin``.
        """
        layout = self.extract_layout()
        unique_texts = list({span["text"] for span in layout if span["text"].strip()})

        self.console.print(
            f"[bold cyan]Translating {len(unique_texts)} unique strings to '{target_lang}'...[/bold cyan]"
        )

        translations = translator.translate_batch(unique_texts, target_lang)

        for span in layout:
            original = span["text"]
            translated = translations.get(original, original)
            span["translated_text"] = translated

            target_width = span["bbox"][2] - span["bbox"][0]
            span["final_font_size"] = self.calculate_optimal_font_size(
                translated, target_width, span["font_size"]
            )
            span["origin"] = (span["bbox"][0], span["bbox"][3])

        return layout

    def close(self) -> None:
        """Release any open resources."""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main() -> None:
    console = Console()

    parser = argparse.ArgumentParser(
        prog="PDF-Master-Translate",
        description="Extract, translate, and fit-to-box PDF text spans.",
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable translation of extracted text spans.",
    )
    parser.add_argument(
        "--lang",
        default="ru",
        help="Target language code for translation (default: ru).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        console.print(f"[bold red]Error:[/] File not found: {args.pdf_path}")
        sys.exit(1)

    processor = PDFProcessor(args.pdf_path)

    # 1. Extract layout
    layout = processor.extract_layout()
    console.print(f"[bold green]✓[/] Extracted {len(layout)} text spans.")

    # Pretty-print first few spans
    table = Table(title="First 10 Text Spans")
    table.add_column("Page", justify="right", style="cyan")
    table.add_column("Text", style="white", no_wrap=False)
    table.add_column("BBox", style="magenta")
    table.add_column("Font", style="green")
    table.add_column("Size", justify="right", style="yellow")

    for span in layout[:10]:
        bbox_str = f"({span['bbox'][0]:.1f}, {span['bbox'][1]:.1f}, {span['bbox'][2]:.1f}, {span['bbox'][3]:.1f})"
        table.add_row(
            str(span["page"] + 1),
            span["text"][:80],
            bbox_str,
            span["font_name"],
            f"{span['font_size']:.1f}",
        )
    console.print(table)

    # 2. Optional translation + font scaling
    if args.translate:
        translator = GoogleTranslator()
        layout = processor.process_translation(translator, args.lang)

        trans_table = Table(title=f"Translations & Scaling (target: {args.lang})")
        trans_table.add_column("Original", style="white", no_wrap=False)
        trans_table.add_column("Translated", style="cyan", no_wrap=False)
        trans_table.add_column("Orig Size", justify="right", style="yellow")
        trans_table.add_column("Final Size", justify="right", style="green")
        for span in layout[:10]:
            trans_table.add_row(
                span["text"][:50],
                span.get("translated_text", "")[:50],
                f"{span['font_size']:.1f}",
                f"{span.get('final_font_size', span['font_size']):.1f}",
            )
        console.print(trans_table)

    # 3. Generate debug visualization
    debug_pdf = processor.visualize_layout(layout=layout)
    console.print(f"[bold green]✓[/] Debug PDF saved to: [cyan]{debug_pdf}[/]")

    processor.close()


if __name__ == "__main__":
    main()
