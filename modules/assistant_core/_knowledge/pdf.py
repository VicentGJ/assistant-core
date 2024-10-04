from pathlib import Path
from typing import List, Iterator
from unittest import loader

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from assistant_core._knowledge.base import AssistantKnowledge


class PDFKnowledgeBase(AssistantKnowledge):
    path: str | Path
    loader: PyPDFLoader

    @classmethod
    def from_path(cls, path: str | Path) -> 'PDFKnowledgeBase':
        """
        Factory method to create a PDFKnowledgeBase instance from a given path.

        Args:
            path (Union[str, Path]): Path to a PDF file or directory containing PDF files.

        Returns:
            PDFKnowledgeBase: An instance of PDFKnowledgeBase with initialized reader.
        """
        _path = Path(path) if isinstance(path, str) else path
        loader = PyPDFLoader(str(_path))
        return cls(path=_path, loader=loader)

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        """Iterate over PDFs and yield lists of documents.
        Each object yielded by the iterator is a list of documents.

        Returns:
            Iterator[List[Document]]: Iterator yielding list of documents
        """

        _pdf_path: Path = Path(self.path) if isinstance(
            self.path, str) else self.path

        if _pdf_path.exists() and _pdf_path.is_dir():
            for _pdf in _pdf_path.glob("**/*.pdf"):
                yield self.loader.read(pdf=_pdf)
        elif _pdf_path.exists() and _pdf_path.is_file() and _pdf_path.suffix == ".pdf":
            yield self.loader.read(pdf=_pdf_path)
