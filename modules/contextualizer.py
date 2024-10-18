from asyncio import gather
from typing import cast

from langchain.chains.base import Chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI

from settings import settings

llm = ChatOpenAI(
    model="radiance",
    base_url=settings.plataformia_base_url,
    api_key=settings.plataformia_api_key,
)
template = """
    <document>
    {doc_content}
    </document>

    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context, in the same language as the original chunk, to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
    Answer only with the succinct context and nothing else. 
"""
custom_contextualizer_prompt = PromptTemplate.from_template(template)
runnable_serializer: RunnableSerializable = (
    {"doc_content": RunnablePassthrough(), "chunk_content": RunnablePassthrough()}
    | custom_contextualizer_prompt
    | llm
    | StrOutputParser()
)
rag_chain = cast(Chain, runnable_serializer)


async def contextualize_chunk(documents: list[Document], chunk: Document) -> Document:
    chunk_content = chunk.page_content
    chunk_source = chunk.metadata["name"]
    matching_doc = next(
        (doc for doc in documents if doc.metadata["name"] == chunk_source and chunk_content in doc.page_content),
        None,
    )

    if matching_doc:
        doc_content = matching_doc.page_content
        context = await rag_chain.ainvoke({"doc_content": doc_content, "chunk_content": chunk_content})
        print(chunk_content)
        contextualized_content = (
            f"<doc_context> {str(context)} <doc_context> \n <page_content> {chunk_content} <page_content>"
        )
        contextualized_chunk = Document(page_content=contextualized_content, metadata=chunk.metadata.copy())
        print(contextualized_chunk)

        return contextualized_chunk
    else:
        raise Exception(f"No matching document found for chunk: {chunk_content[:50]}...")


async def get_contextualized_chunks(docs_splits: list[Document], chunks: list[Document]):
    tasks = [contextualize_chunk(docs_splits, chunk) for chunk in chunks]
    contextualized_chunks = await gather(*tasks)

    return [chunk for chunk in contextualized_chunks if chunk is not None]
