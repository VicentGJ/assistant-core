from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS
from tqdm import tqdm
from utils.ollama import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools.base import BaseTool

import os


def load_existing_index(
    vectors_path: str, index_name: str, embedding: Embeddings
) -> FAISS | None:
    full_path = os.path.join(vectors_path, f"{index_name}.faiss")
    if os.path.exists(full_path):
        return FAISS.load_local(
            vectors_path, allow_dangerous_deserialization=True, embeddings=embedding
        )
    return None


def create_faiss_index(docs, embedding: Embeddings, batch_size: int = 20) -> FAISS:
    db = None
    for i in tqdm(range(0, len(docs), batch_size), desc="Processing docs"):
        batch = docs[i : i + batch_size]
        if db is None:
            db = FAISS.from_documents(documents=batch, embedding=embedding)
        else:
            db.add_documents(documents=batch)
    return db


def get_faiss(
    data_path: str,
    vectors_path: str,
    index_name: str = "index",
    embedding: Embeddings | None = None,
    recreate: bool = False,
) -> FAISS:
    if embedding is None:
        embedding = OpenAIEmbeddings()

    if not recreate:
        existing_index = load_existing_index(vectors_path, index_name, embedding)
        if existing_index:
            return existing_index

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load_and_split(text_splitter)

    db = create_faiss_index(docs, embedding)
    db.save_local(vectors_path, index_name)
    return db


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    custom_description: str | None = None
    knowledge_base: VectorStore

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

            results = self.knowledge_base.similarity_search(
                query=query, k=num_documents
            )

            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(f"Document {i}:\n{doc.page_content}\n")

            return "\n".join(formatted_results)

        except Exception as e:
            print(e)
            return f"An error occurred: {str(e)}"
