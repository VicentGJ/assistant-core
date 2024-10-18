from abc import ABC, abstractmethod
from asyncio import run
from os.path import exists, join
from typing import Any
from uuid import uuid4

from faiss import IndexFlatL2
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_cohere import CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import RetrieverLike
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

from modules.contextualizer import get_contextualized_chunks
from settings import settings


class VectorizerInterface(VectorStore, ABC):
    def send_docs_to_vectorstore(self, docs: list[Document], contextualize_docs: bool = False):
        try:
            docs = self._combine_documents(doc_list=docs)
            docs_chunks = []
            if not contextualize_docs:
                docs_chunks = self._split_docs(docs=docs, tokens_per_chunk=250, chunk_overlap=100)
            else:
                docs_splits = self._split_docs(docs=docs)
                chunks = self._split_docs(docs_splits, 250, 100)
                docs_chunks = run(get_contextualized_chunks(docs_splits, chunks))
            uuids = [str(uuid4()) for _ in range(len(docs_chunks))]

            print("SENDING DOCS TO VECTOR STORE...")
            self.add_documents(documents=docs_chunks, ids=uuids)
            print("DOCS SENT TO VECTOR STORE.")
        except Exception as e:
            raise Exception(f"Error sending to vector store: {e}")

    @abstractmethod
    def delete_all_vectors(self, **kwargs: Any):
        pass

    def search_similar_docs(self, query: str) -> list[Document]:
        try:
            docs = self.similarity_search(
                query=query,
                k=5,
            )
            return self._combine_documents(docs)
        except Exception as e:
            raise Exception(f"Error doing similarity search: {e}")

    @abstractmethod
    def hybrid_search(self, query: str, amount_of_docs: int = 5, use_bm25_search: bool = False) -> list[Document]:
        pass

    @staticmethod
    def _split_docs(docs: list[Document], tokens_per_chunk: int = 32000, chunk_overlap: int = 0) -> list[Document]:
        text_splitter = TokenTextSplitter(
            chunk_size=tokens_per_chunk,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(docs)
        return splits

    @staticmethod
    def _combine_documents(doc_list: list[Document]) -> list[Document]:
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
        embeddings: Embeddings = OpenAIEmbeddings(
            model="embeddings",
            openai_api_base=settings.plataformia_base_url,
            openai_api_key=settings.plataformia_api_key,
        ),
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

    def send_docs_to_vectorstore(self, docs: list[Document], contextualize_docs: bool = False):
        super().send_docs_to_vectorstore(docs, contextualize_docs)
        self.save_local(folder_path=self.vectors_path, index_name=self.index_name)

    def delete_all_vectors(self):
        ids = self.index_to_docstore_id.values()
        self.delete(ids)

    def hybrid_search(
        self, query: str, amount_of_relevant_docs: int = 5, use_bm25_search: bool = False
    ) -> list[Document]:
        similar_docs = self.similarity_search(query=query, k=amount_of_relevant_docs * 2)

        if use_bm25_search == True:
            bm25_docs = self._bm25_search(query=query, bm25_k=amount_of_relevant_docs)
            similar_docs.extend(bm25_docs)

        reranker = CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model="rerank-multilingual-v3.0",
            top_n=amount_of_relevant_docs,
        )

        compressed_docs = reranker.compress_documents(documents=similar_docs, query=query)

        return list(compressed_docs)

    def _bm25_search(self, query: str, bm25_k: int = 5) -> list[Document]:
        index_size = len(self.index_to_docstore_id)
        retriever = self.as_retriever(search_kwargs={"k": index_size})
        docs = retriever.invoke("")
        bm25retriever = BM25Retriever.from_documents(documents=docs, k=bm25_k)
        results = bm25retriever.invoke(query)

        return results
