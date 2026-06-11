"""Loaders that turn raw text and files on disk into :class:`Document` objects."""

from __future__ import annotations

from pathlib import Path

from ragforge.types import Document

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".log"}


def load_text(text: str, *, doc_id: str, metadata: dict[str, str] | None = None) -> Document:
    """Wrap an in-memory string as a :class:`Document`."""
    return Document(id=doc_id, text=text, metadata=metadata or {})


def load_path(path: str | Path) -> list[Document]:
    """Load a text file or a directory tree of text files as documents.

    Directories are walked recursively; only files with a recognised text
    suffix are read. Each document's id is its path relative to the root.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such path: {root}")

    if root.is_file():
        return [_read_file(root, base=root.parent)]

    docs: list[Document] = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in _TEXT_SUFFIXES:
            docs.append(_read_file(file, base=root))
    return docs


def _read_file(file: Path, *, base: Path) -> Document:
    rel = file.relative_to(base).as_posix()
    return Document(
        id=rel,
        text=file.read_text(encoding="utf-8", errors="replace"),
        metadata={"source": str(file), "suffix": file.suffix.lower()},
    )
