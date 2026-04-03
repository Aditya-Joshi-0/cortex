"""
Cortex RAG — Document Loader
Handles PDF, HTML, and plain-text ingestion.
Returns a list of Document dataclasses ready for chunking.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Raw document before chunking."""
    doc_id: str                          # sha256 of source path
    source: str                          # original file path / URL
    doc_type: str                        # "pdf" | "html" | "text"
    title: str
    text: str                            # full cleaned text
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()[:16]


class DocumentLoader:
    """
    Load documents from disk.

    Supports:
      - PDF  → pdfplumber (better layout) with PyPDF2 fallback
      - HTML → BeautifulSoup main-content extraction
      - TXT  → direct read with encoding detection
    """

    def __init__(self) -> None:
        self._loaders = {
            ".pdf":  self._load_pdf,
            ".html": self._load_html,
            ".htm":  self._load_html,
            ".txt":  self._load_text,
            ".md":   self._load_text,
        }

    # ── Public ────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> Document:
        """Load a single file and return a Document."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        loader = self._loaders.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info("Loading %s (%s)", path.name, suffix)
        return loader(path)

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Document]:
        """Load all supported files from a directory."""
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        docs: list[Document] = []
        for path in directory.glob(pattern):
            if path.suffix.lower() in self._loaders and path.is_file():
                try:
                    docs.append(self.load_file(path))
                except Exception as exc:
                    logger.warning("Skipping %s — %s", path, exc)
        logger.info("Loaded %d documents from %s", len(docs), directory)
        return docs

    # ── Private loaders ───────────────────────────────────────

    def _load_pdf(self, path: Path) -> Document:
        text = self._extract_pdf_text(path)
        return Document(
            doc_id=Document.make_id(str(path)),
            source=str(path),
            doc_type="pdf",
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            text=self._clean_text(text),
            metadata={"filename": path.name, "pages": text.count("\f") + 1},
        )

    def _load_html(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, title = self._extract_html_content(raw)
        return Document(
            doc_id=Document.make_id(str(path)),
            source=str(path),
            doc_type="html",
            title=title or path.stem,
            text=self._clean_text(text),
            metadata={"filename": path.name},
        )

    def _load_text(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            doc_id=Document.make_id(str(path)),
            source=str(path),
            doc_type="text",
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            text=self._clean_text(raw),
            metadata={"filename": path.name},
        )

    # ── Text extraction helpers ────────────────────────────────

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Try pdfplumber first, fall back to PyPDF2."""
        try:
            import pdfplumber  # type: ignore
            pages: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
            return "\n\n".join(pages)
        except ImportError:
            pass

        try:
            import PyPDF2  # type: ignore
            pages = []
            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
            return "\n\n".join(pages)
        except ImportError as exc:
            raise RuntimeError(
                "Install pdfplumber or PyPDF2: pip install pdfplumber"
            ) from exc

    @staticmethod
    def _extract_html_content(html: str) -> tuple[str, Optional[str]]:
        """Extract main text content and title from HTML."""
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install beautifulsoup4: pip install beautifulsoup4"
            ) from exc

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Remove boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer <main> or <article>, fall back to <body>
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = (main or soup).get_text(separator="\n", strip=True)
        return text, title

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace, remove null bytes and common PDF artefacts."""
        text = text.replace("\x00", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove lone hyphenation artefacts from PDF line-breaks
        text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
        return text.strip()
