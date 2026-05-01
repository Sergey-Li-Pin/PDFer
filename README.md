# PDF-Master-Translate

A lightweight Python CLI tool for parsing and translating PDF documents. Built on top of [PyMuPDF](https://github.com/pymupdf/PyMuPDF), it extracts text while preserving layout and integrates seamlessly with translation workflows.

## Features

- Extract clean text from PDF files
- Preserve document structure and layout
- Easy-to-use programmatic API
- Ready for future translation pipeline integration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-master-translate.git
   cd pdf-master-translate
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```python
from src.parser import PDFProcessor

# Initialize the processor with a PDF file
processor = PDFProcessor("document.pdf")

# Extract text from all pages
text = processor.extract_text()
print(text)
```

## License

This project is licensed under the MIT License.