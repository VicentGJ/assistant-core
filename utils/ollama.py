from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from ollama import AsyncClient, Client


class OllamaEmbeddings(BaseModel, Embeddings):
    """OllamaEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaEmbeddings

            model = OllamaEmbeddings(model="llama3")
            embedder.embed_query("what is the place that jonathan worked at?")
    """

    model: str = "llama2"
    """Model name to use."""
    host: str = None
    """Host to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = []
        for doc in texts:
            embedded_docs.append(
                list(Client(host=self.host).embeddings(self.model, doc)["embedding"]))
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = []
        for doc in texts:
            embedded_docs.append(
                list((await AsyncClient(host=self.host).embeddings(self.model, doc))["embedding"])
            )
        return embedded_docs

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
