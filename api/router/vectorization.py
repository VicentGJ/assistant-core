from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from api.dependencies import check_connector_credentials
from modules.connectors import ConnectorInterface
from modules.db.interfaces import SQLAlchemyDatabase
from modules.db.managers import DatabaseManager
from modules.vectorizers import FaissVectorizer

router = APIRouter(prefix="/vectorization", tags=["vectorization"])


@router.post("/{storage_bucket_id}")
async def vectorize_nextcloud_docs(
    connector: ConnectorInterface = Depends(check_connector_credentials),
):
    def event_stream():
        try:
            start_time = time()
            db_manager = DatabaseManager(SQLAlchemyDatabase())
            vectorstore = FaissVectorizer(index_name=connector.username)

            db_manager.register_user(connector.username)

            # Use this commented code to delete everything and start over
            # db_manager.delete_all_documents_by_username(connector.username)
            # vectorstore.delete_all_vectors()

            files = connector.get_files(db_manager)
            total_elements = len(files)
            processed_elements = 0

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(
                        connector.process_file, file, vectorstore, db_manager
                    )
                    for file in files
                ]

                for future in as_completed(futures):
                    file_name = future.result()
                    processed_elements += 1
                    progress = (processed_elements / total_elements) * 100
                    if file_name:
                        yield (
                            f'data: {{ "status": 200, "message": "Processed element '
                            f'{file_name}.", "progress": {progress:.2f} }}\n\n'
                        )
            duration = time() - start_time
            yield f'data: {{ "status": 200, "message": "Finished in {duration:.4f} seconds." }}\n\n'

        except Exception as e:
            yield f'data: {{"status": 500, "message": "Error: {e}" }}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
