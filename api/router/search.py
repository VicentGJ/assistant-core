from fastapi import APIRouter, Depends, Query
from starlette.responses import JSONResponse

from api.dependencies import check_connector_credentials
from modules.connectors import BaseConnector
from modules.vectorizers import FaissVectorizer

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
async def search_similarity_docs(
    client: BaseConnector = Depends(check_connector_credentials),
    query: str = Query(...),
):
    try:
        vectorstore = FaissVectorizer(index_name=client.username)
        docs = vectorstore.search_similarity(query=query)

        # Convert the list of Document objects to dictionary representation
        serialized_docs = [doc.__dict__ for doc in docs]

        return serialized_docs

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error: {e}"},
        )
