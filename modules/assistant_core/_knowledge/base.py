from langchain.tools import BaseTool
import json
from pydoc import doc
from typing import Iterator
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class AssistantKnowledge(BaseModel):
    loader: BaseLoader | None = None
    vector_db: VectorStore | None = None
    num_documents: int = 0

    @property
    def document_lists(self) -> Iterator[list[Document]]:
        """Iterator that yields lists of documents in the knowledge base
        Each object yielded by the iterator is a list of documents.
        """
        raise NotImplementedError

    def search(self, query: str, num_documents: int | None = None) -> list[str]:
        """ "Returns relevant documents matching the query"""

        _num_documents = num_documents or self.num_documents

        return self.vector_db.similarity_search(query, k=_num_documents)

    def load(
        self, recreate: bool = False, upsert: bool = False, skip_existing: bool = True
    ) -> None:

        if self.vector_db is None:
            raise Exception("No vectorDB provided")

        num_documents = 0
        for document_list in self.document_lists:
            documents_to_load = document_list
            num_documents += len(documents_to_load)
            docs = self.loader.load_and_split(documents_to_load)
            self.vector_db.add_documents(docs)

    def load_documents(
        self,
        documents: list[Document],
        recreate: bool = False,
        upsert: bool = False,
        skip_existing: bool = True,
    ) -> None:
        if self.vector_db is None:
            raise Exception("No vectorDB provided")

        docs = RecursiveCharacterTextSplitter().split_documents(documents)
        self.vector_db.add_documents(docs)

    def load_document(
        self,
        document: Document,
        recreate: bool = False,
        upsert: bool = False,
        skip_existing: bool = True,
    ) -> None:
        if self.vector_db is None:
            raise Exception("No vectorDB provided")

        docs = RecursiveCharacterTextSplitter().split_documents([document])
        self.vector_db.add_documents(docs)

    def load_text(
        self,
        text: str,
        recreate: bool = False,
        upsert: bool = False,
        skip_existing: bool = True,
    ) -> None:
        if self.vector_db is None:
            raise Exception("No vectorDB provided")

        docs = RecursiveCharacterTextSplitter().split_text(text)
        self.vector_db.add_documents(docs)


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    custom_description: str | None = None
    knowledge_base: AssistantKnowledge = Field(
        ..., description="The assistant's knowledge base"
    )

    @property
    def description(self) -> str:

        description_template = """
        Args:
            query (str): The search query.
            num_documents (int, optional): Number of documents to return. Defaults to 5.
        
        Returns:
            str: A list of relevant documents matching the query.
        """

        if self.custom_description:
            return self.custom_description + "\n" + description_template
        return (
            "Use this tool to perform similarity searches on the assistant's knowledge base.\n"
            + description_template
        )

    def _run(self, query: str, num_documents: int = 5) -> str:
        try:
            if not query:
                return "Invalid input. 'query' is required."

            results = self.knowledge_base.search(query, num_documents)

            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(f"Document {i}:\n{doc.page_content}\n")

            return "\n".join(formatted_results)

        except Exception as e:
            print(e)
            return f"An error occurred: {str(e)}"
