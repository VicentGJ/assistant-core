from abc import ABC, abstractmethod
from asyncio import run
from os.path import join, exists
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from faiss import IndexFlatL2
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from modules.contextualizer import get_contextualized_chunks
from settings import settings


class VectorizerInterface(VectorStore, ABC):
    def send_docs_to_vectorstore(
        self, docs: list[Document], contextualize_docs: bool = False
    ):
        print("SENDING DOCS TO VECTOR STORE...")
        try:
            docs_chunks = []
            if not contextualize_docs:
                docs_chunks = self.split_docs(
                    docs=docs, chunk_size=500, chunk_overlap=200
                )
            else:
                docs_splits = self.split_docs(docs=docs)
                chunks = self.split_docs(docs_splits, 60, 15)
                docs_chunks = run(get_contextualized_chunks(docs_splits, chunks))
            uuids = [str(uuid4()) for _ in range(len(docs_chunks))]
            self.add_documents(documents=docs_chunks, ids=uuids)
            print("DOCS SENT TO VECTOR STORE.")
        except Exception as e:
            raise Exception(f"Error sending to vector store: {e}")

    @abstractmethod
    def delete_all_vectors(self, **kwargs: Any):
        pass

    def search_similarity(self, query):
        try:
            docs = self.similarity_search(
                query=query,
                k=5,
            )
            return self.combine_documents(docs)
        except Exception as e:
            raise Exception(f"Error doing similarity search: {e}")

    @staticmethod
    def split_docs(
        docs: list[Document], chunk_size: int = 32000, chunk_overlap: int = 0
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        splits = text_splitter.split_documents(docs)
        return splits

    @staticmethod
    def combine_documents(doc_list):
        combined_content = {}
        processed_sources = set()

        for doc in doc_list:
            source = doc.metadata["source"]
            if source not in processed_sources:
                processed_sources.add(source)
                combined_content[source] = doc.page_content
            else:
                combined_content[source] += "\n\n" + doc.page_content
                doc.page_content = ""

        modified_docs = []
        for doc in doc_list:
            if doc.page_content != "":
                doc.page_content = combined_content[doc.metadata["source"]]
                modified_docs.append(doc)

        return modified_docs


class FaissVectorizer(VectorizerInterface, FAISS):
    def __init__(
        self,
        index_name: str,
        embeddings: Embeddings = OpenAIEmbeddings(),
        vectors_path: str = settings.faiss_folder_path,
    ):
        try:
            self.index_name = index_name
            self.vectors_path = vectors_path
            faiss_instance = self._load_existing_faiss_instance(embeddings)

            if not faiss_instance:
                faiss_instance = FAISS(
                    embedding_function=embeddings,
                    index=IndexFlatL2(len(embeddings.embed_query("hello world"))),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
            self.__dict__.update(faiss_instance.__dict__)
        except Exception as e:
            raise Exception(f"Error instanciating FAISS vector store: {e}")

    def _load_existing_faiss_instance(self, embedding: Embeddings) -> FAISS | None:
        full_path = join(self.vectors_path, f"{self.index_name}.faiss")

        if exists(full_path):
            return FAISS.load_local(
                folder_path=self.vectors_path,
                index_name=self.index_name,
                embeddings=embedding,
                allow_dangerous_deserialization=True,
            )
        return None

    def send_docs_to_vectorstore(
        self, docs: list[Document], contextualize_docs: bool = False
    ):
        super().send_docs_to_vectorstore(docs, contextualize_docs)
        self.save_local(folder_path=self.vectors_path, index_name=self.index_name)

    def delete_all_vectors(self):
        ids = self.index_to_docstore_id.values()
        self.delete(ids)
