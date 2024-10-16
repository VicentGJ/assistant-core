from fastapi import APIRouter, Depends, Query
from starlette.responses import JSONResponse

from api.dependencies import check_connector_credentials, validate_token
from modules.vectorizers import FaissVectorizer


router = APIRouter(prefix="/search", tags=["search"])


@router.get(
    "/{storage_bucket_name}",
    dependencies=[Depends(validate_token), Depends(check_connector_credentials)],
)
async def search_similarity_docs(
    storage_bucket_name: str,
    query: str = Query(...),
):
    try:
        vectorstore = FaissVectorizer(index_name=storage_bucket_name)

        # docs = vectorstore.search_similarity(query=query)

        # # Convert the list of Document objects to dictionary representation
        # serialized_docs = [doc.__dict__ for doc in docs]

        # return serialized_docs
        return vectorstore.hybrid_search(query, use_bm25_search=True)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error: {e}"},
        )
