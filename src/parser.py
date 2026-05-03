import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from font_manager import FontManager
from translator import BaseTranslator, GoogleTranslator, OllamaTranslator, OpenRouterTranslator

DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "DejaVuSans.ttf"
)


class PDFProcessor:
    """
    Extract text blocks with coordinates and styles from PDF documents,
    generate visual debug output, process translations, fit translated
    text to original bounding boxes, and reconstruct the final PDF.
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
        self.font_path_abs = os.path.abspath(self.font_path)
        self.console = Console()
        self._doc: fitz.Document | None = None
        self._font: fitz.Font | None = None

    def _open(self) -> fitz.Document:
        """Lazy-open the PDF document."""
        if self._doc is None:
            self._doc = fitz.open(self.file_path)
        return self._doc

    def _get_font(self, font_path: str | None = None) -> fitz.Font:
        """Lazy-load an external TTF font for width calculations."""
        path = font_path or self.font_path_abs
        # Simple cache-per-path could be added, but for now we create fresh
        return fitz.Font(fontfile=path)

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
            - block: block index within the page
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
                for block_idx, block in enumerate(page_dict.get("blocks", [])):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            spans.append(
                                {
                                    "page": page_num,
                                    "block": block_idx,
                                    "text": span.get("text", ""),
                                    "bbox": tuple(span["bbox"]),
                                    "font_size": span.get("size", 0.0),
                                    "font_name": span.get("font", ""),
                                }
                            )
                progress.advance(task)

        return spans

    @staticmethod
    def _distribute_text(text: str, weights: list[float]) -> list[str]:
        """
        Split ``text`` into pieces proportionally to ``weights``.

        Args:
            text: The full translated paragraph.
            weights: Relative weights (e.g. original span text lengths).

        Returns:
            List of text fragments matching the length of ``weights``.
        """
        words = text.split()
        if not words or sum(weights) == 0:
            return [text] + [""] * (len(weights) - 1)

        total_weight = sum(weights)
        counts = []
        allocated = 0
        for i, w in enumerate(weights[:-1]):
            n = round(len(words) * w / total_weight)
            counts.append(max(0, n))
            allocated += n
        counts.append(max(0, len(words) - allocated))

        result = []
        idx = 0
        for c in counts:
            result.append(" ".join(words[idx : idx + c]))
            idx += c
        return result

    def calculate_optimal_font_size(
        self,
        text: str,
        target_width: float,
        original_font_size: float,
        font_path: str | None = None,
    ) -> float:
        """
        Dynamically reduce font size until ``text`` fits inside ``target_width``.

        Args:
            text: The (translated) text to measure.
            target_width: Available width in points (bbox width).
            original_font_size: Starting font size.
            font_path: Optional TTF path override.

        Returns:
            The largest font size ≤ ``original_font_size`` that fits,
            or ``6.0`` as a hard floor.
        """
        font = self._get_font(font_path)
        font_size = original_font_size
        MIN_SIZE = 6.0

        while font_size > MIN_SIZE:
            measured = font.text_length(text, fontsize=font_size)
            if measured <= target_width:
                break
            font_size -= 0.5

        return max(font_size, MIN_SIZE)

    def visualize_layout(
        self,
        output_path: str | None = None,
        layout: list[dict] | None = None,
        target_lang: str | None = None,
    ) -> str:
        """
        Create a copy of the PDF with red rectangles around every text span.
        If ``layout`` contains translated text and ``final_font_size``,
        ghost-prints the translated string inside the box in blue using
        ``fitz.TextWriter`` for reliable font embedding.

        Args:
            output_path: Destination file path. Defaults to ``output/debug_layout.pdf``.
            layout: Pre-computed layout list (e.g. from ``process_translation``).
                    If ``None``, ``extract_layout()`` is called.
            target_lang: Target language code; forces DejaVuSans for ``'ru'``.

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
                    font_path = span.get("mapped_font_path", self.font_path_abs)
                    font = fitz.Font(fontfile=font_path)
                    tw = fitz.TextWriter(page.rect)
                    tw.append(
                        origin,
                        span["translated_text"],
                        font=font,
                        fontsize=span["final_font_size"],
                    )
                    tw.write_text(page, color=(0, 0, 1), render_mode=0)
                progress.advance(task)

        debug_doc.save(output_path)
        debug_doc.close()
        return output_path

    def process_translation(
        self,
        translator: BaseTranslator,
        target_lang: str,
        font_manager: FontManager | None = None,
        threads: int = 4,
        hybrid: bool = False,
    ) -> list[dict]:
        """
        Translate text in paragraph mode, map words back to original spans,
        compute ``final_font_size``, and attach font mapping.

        Args:
            translator: An instance of a ``BaseTranslator`` subclass.
            target_lang: Target language code (e.g. 'ru', 'en').
            font_manager: Optional ``FontManager`` for style-aware fonts.

        Returns:
            The layout list with added fields:
            ``translated_text``, ``final_font_size``, ``origin``, and
            ``mapped_font_path``.
        """
        layout = self.extract_layout()

        # Group spans by (page, block) to form paragraphs
        blocks = defaultdict(list)
        for span in layout:
            blocks[(span["page"], span["block"])].append(span)

        # Build paragraph texts
        paragraph_texts = []
        block_keys = []
        for key in sorted(blocks.keys()):
            spans = blocks[key]
            para_text = " ".join(s["text"] for s in spans)
            paragraph_texts.append(para_text)
            block_keys.append(key)

        self.console.print(
            f"[bold cyan]Translating {len(paragraph_texts)} paragraphs to '{target_lang}'...[/bold cyan]"
        )

        if hybrid:
            # Stage 1: Google draft (parallel)
            self.console.print(
                "[bold cyan]Stage 1: Generating Google draft translations...[/bold cyan]"
            )
            google_translator = GoogleTranslator()
            draft_results = google_translator.translate_batch(
                paragraph_texts, target_lang, threads=threads
            )

            # Stage 2: LLM polish
            if isinstance(translator, OllamaTranslator):
                self.console.print(
                    "[bold cyan]Stage 2: Polishing translations with Ollama (sequential)...[/bold cyan]"
                )
                polished_results: dict[str, str] = {}
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Polishing translations...", total=len(paragraph_texts)
                    )
                    for para_text in paragraph_texts:
                        draft = draft_results.get(para_text, para_text)
                        polished = translator.translate(
                            para_text, target_lang, draft_text=draft
                        )
                        polished_results[para_text] = polished
                        progress.advance(task)
                translations = polished_results
            else:
                # OpenRouter and other cloud APIs can handle parallel requests
                self.console.print(
                    "[bold cyan]Stage 2: Polishing translations with OpenRouter (parallel)...[/bold cyan]"
                )
                polished_results = {}
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Polishing translations...", total=len(paragraph_texts)
                    )
                    with ThreadPoolExecutor(max_workers=threads) as executor:
                        futures = {
                            executor.submit(
                                translator.translate,
                                para_text,
                                target_lang,
                                draft_results.get(para_text, para_text),
                            ): para_text
                            for para_text in paragraph_texts
                        }
                        for future in as_completed(futures):
                            para_text = futures[future]
                            try:
                                polished = future.result()
                            except Exception:  # noqa: BLE001
                                polished = draft_results.get(para_text, para_text)
                            polished_results[para_text] = polished
                            progress.advance(task)
                translations = polished_results
        else:
            # Ollama local LLMs struggle with parallel generation on CPU,
            # so we force sequential processing for that engine.
            if isinstance(translator, OllamaTranslator):
                batch_threads = 1
            else:
                batch_threads = threads

            translations = translator.translate_batch(
                paragraph_texts, target_lang, threads=batch_threads
            )

        # Map translated text back to individual spans
        for key, para_text in zip(block_keys, paragraph_texts):
            spans = blocks[key]
            translated_para = translations.get(para_text, para_text)

            weights = [len(s["text"]) for s in spans]
            distributed = self._distribute_text(translated_para, weights)

            for span, translated_text in zip(spans, distributed):
                span["translated_text"] = translated_text

                target_width = span["bbox"][2] - span["bbox"][0]
                mapped_font = (
                    font_manager.get_font_path(span["font_name"])
                    if font_manager
                    else self.font_path_abs
                )
                span["mapped_font_path"] = mapped_font
                span["final_font_size"] = self.calculate_optimal_font_size(
                    translated_text, target_width, span["font_size"], mapped_font
                )
                span["origin"] = (span["bbox"][0], span["bbox"][3])

        return layout

    def reconstruct_pdf(
        self,
        layout: list[dict],
        output_path: str | None = None,
        target_lang: str | None = None,
    ) -> str:
        """
        Generate the final translated PDF.

        Steps:
        1. Open a fresh copy of the original PDF.
        2. Add redaction annotations for every original text span and apply
           them (with ``PDF_REDACT_IMAGE_NONE``) to clear text without
           touching images.
        3. Render ``translated_text`` via ``fitz.TextWriter`` for reliable
           font embedding, using per-span mapped fonts.
        4. Preserve original metadata and save.

        Args:
            layout: Layout list produced by ``process_translation``.
            output_path: Destination file path.
                         Defaults to ``output/translated_final.pdf``.
            target_lang: Target language code; forces DejaVuSans for ``'ru'``.

        Returns:
            Path to the reconstructed PDF.
        """
        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "output"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "translated_final.pdf")

        self.console.print(
            f"[bold cyan]Using font:[/] {self.font_path_abs}"
        )

        src_doc = self._open()
        new_doc = fitz.open(self.file_path)

        # Preserve metadata
        new_doc.metadata = src_doc.metadata

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            # 1. Mark original text spans for redaction
            redact_task = progress.add_task(
                "Marking original text for removal...", total=len(layout)
            )
            for span in layout:
                page = new_doc[span["page"]]
                page.add_redact_annot(fitz.Rect(span["bbox"]))
                progress.advance(redact_task)

            # 2. Apply redactions page by page (keep images intact)
            apply_task = progress.add_task(
                "Applying redactions...", total=len(new_doc)
            )
            for page in new_doc:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                progress.advance(apply_task)

            # 3. Render translated text with TextWriter for correct embedding
            render_task = progress.add_task(
                "Rendering translated text...", total=len(layout)
            )
            for span in layout:
                page = new_doc[span["page"]]
                origin = span.get("origin", (span["bbox"][0], span["bbox"][3]))
                font_path = span.get("mapped_font_path", self.font_path_abs)
                font = fitz.Font(fontfile=font_path)
                tw = fitz.TextWriter(page.rect)
                tw.append(
                    origin,
                    span["translated_text"],
                    font=font,
                    fontsize=span["final_font_size"],
                )
                tw.write_text(page, color=(0, 0, 0), render_mode=0)
                progress.advance(render_task)

        new_doc.save(output_path)
        new_doc.close()
        return output_path

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
        description="Extract, translate, fit-to-box, and reconstruct PDF text spans.",
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
    parser.add_argument(
        "--engine",
        default="google",
        choices=["google", "ollama", "openrouter"],
        help="Translation engine to use (default: google).",
    )
    parser.add_argument(
        "--font-config",
        default=None,
        help="Path to font_map.json (default: project root font_map.json).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel translation workers (default: 4, max: 16).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "Enable two-stage hybrid translation: Google draft + LLM polish. "
            "Uses Ollama when --engine ollama, otherwise OpenRouter."
        ),
    )
    args = parser.parse_args()
    if args.threads < 1:
        args.threads = 1
    elif args.threads > 16:
        args.threads = 16

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
        if args.hybrid:
            if args.engine == "ollama":
                translator = OllamaTranslator()
            else:
                translator = OpenRouterTranslator()
        elif args.engine == "ollama":
            translator = OllamaTranslator()
        elif args.engine == "openrouter":
            translator = OpenRouterTranslator()
        else:
            translator = GoogleTranslator()

        font_manager = FontManager(args.font_config)
        report = font_manager.style_report()
        console.print("[bold cyan]Font mapping:[/]")
        for style, path in report.items():
            console.print(f"  {style}: [green]{os.path.basename(path)}[/]")

        layout = processor.process_translation(
            translator,
            args.lang,
            font_manager,
            threads=args.threads,
            hybrid=args.hybrid,
        )

        trans_table = Table(title=f"Translations & Scaling (target: {args.lang})")
        trans_table.add_column("Original", style="white", no_wrap=False)
        trans_table.add_column("Translated", style="cyan", no_wrap=False)
        trans_table.add_column("Orig Size", justify="right", style="yellow")
        trans_table.add_column("Final Size", justify="right", style="green")
        trans_table.add_column("Mapped Font", style="magenta")
        for span in layout[:10]:
            trans_table.add_row(
                span["text"][:40],
                span.get("translated_text", "")[:40],
                f"{span['font_size']:.1f}",
                f"{span.get('final_font_size', span['font_size']):.1f}",
                os.path.basename(span.get("mapped_font_path", "")),
            )
        console.print(trans_table)

        # 3. Reconstruct the final translated PDF
        final_pdf = processor.reconstruct_pdf(layout, target_lang=args.lang)
        console.print(
            f"[bold green]✓[/] Final translated PDF saved to: [cyan]{final_pdf}[/]"
        )

    # 4. Generate debug visualization
    debug_pdf = processor.visualize_layout(layout=layout, target_lang=args.lang)
    console.print(f"[bold green]✓[/] Debug PDF saved to: [cyan]{debug_pdf}[/]")

    processor.close()


if __name__ == "__main__":
    main()
