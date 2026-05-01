import os
import sys

import fitz
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


class PDFProcessor:
    """
    Extract text blocks with coordinates and styles from PDF documents,
    and generate visual debug output.
    """

    def __init__(self, file_path: str):
        """
        Initialize the PDFProcessor and open the PDF document.

        Args:
            file_path: Path to the PDF file to process.
        """
        self.file_path = file_path
        self.console = Console()
        self._doc: fitz.Document | None = None

    def _open(self) -> fitz.Document:
        """Lazy-open the PDF document."""
        if self._doc is None:
            self._doc = fitz.open(self.file_path)
        return self._doc

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

    def visualize_layout(self, output_path: str | None = None) -> str:
        """
        Create a copy of the PDF with red rectangles drawn around every text span.

        Args:
            output_path: Destination file path. Defaults to output/debug_layout.pdf.

        Returns:
            The path to the generated debug PDF.
        """
        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "output"
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "debug_layout.pdf")

        doc = self._open()
        debug_doc = fitz.open(self.file_path)  # work on a fresh copy
        spans = self.extract_layout()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Drawing bounding boxes...", total=len(spans)
            )
            for span in spans:
                page = debug_doc[span["page"]]
                rect = fitz.Rect(span["bbox"])
                page.draw_rect(rect, color=(1, 0, 0), width=1.5)
                progress.advance(task)

        debug_doc.save(output_path)
        debug_doc.close()
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

    if len(sys.argv) < 2:
        console.print(
            "[bold red]Usage:[/] python src/parser.py <path_to_pdf>",
            style="red",
        )
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        console.print(f"[bold red]Error:[/] File not found: {pdf_path}")
        sys.exit(1)

    processor = PDFProcessor(pdf_path)

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

    # 2. Generate debug visualization
    debug_pdf = processor.visualize_layout()
    console.print(f"[bold green]✓[/] Debug PDF saved to: [cyan]{debug_pdf}[/]")

    processor.close()


if __name__ == "__main__":
    main()
