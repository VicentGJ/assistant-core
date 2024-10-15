from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from settings import settings

llm = ChatOpenAI(
    model="spark",
    base_url=settings.plataformia_base_url,
    api_key=settings.plataformia_api_key,
)
template = """
    <document>
    {doc_content}
    </document>

    Here is the chunk we want to situate within the whole document:
    <chunk>
    {chunk_content}
    </chunk>

    Please provide a short succinct context, to situate this chunk within the overall document for the purposes of improving search retrieval.
    Answer only with the succinct context and nothing else.
"""
custom_contextualizer_prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"doc_content": RunnablePassthrough(), "chunk_content": RunnablePassthrough()}
    | custom_contextualizer_prompt
    | llm
    | StrOutputParser()
)


async def contextualize_chunks(
    documents: list[Document], chunks: list[Document]
) -> list[Document]:
    contextualized_chunks = []

    for chunk in chunks:
        chunk_content = chunk.page_content
        chunk_source = chunk.metadata["source"]

        matching_doc = next(
            (
                doc
                for doc in documents
                if doc.metadata["source"] == chunk_source
                and chunk_content in doc.page_content
            ),
            None,
        )

        if matching_doc:
            doc_content = matching_doc.page_content

            context = await rag_chain.ainvoke(
                {"doc_content": doc_content, "chunk_content": chunk_content}
            )

            contextualized_chunk = Document(
                page_content=context, metadata=chunk.metadata.copy()
            )

            contextualized_chunks.append(contextualized_chunk)
        else:
            print(f"No matching document found for chunk: {chunk_content[:50]}...")
            # No se agrega ningun chuhnk si no hay contexto , creo q si se debe hacer de igual forma

    return contextualized_chunks


async def acontextualization(documents, chunks):
    documents = documents
    chunks = chunks

    contextualized_chunks = await contextualize_chunks(documents, chunks, rag_chain)
    return contextualized_chunks


async def get_contextualized_chunks(docs_splits, chunks):
    # Esperar hasta que la función asincrónica termine y obtenga el resultado
    contextualized_chunks = await acontextualization(docs_splits, chunks)

    # Asegurarse de que contextualized_chunks no esté vacío
    while not contextualized_chunks:
        print("Esperando a que se llene contextualized_chunks...")
        await asyncio.sleep(1)  # Esperar un segundo antes de verificar de nuevo
        contextualized_chunks = await acontextualization(
            docs_splits, chunks
        )  # Intentar nuevamente

    return contextualized_chunks
