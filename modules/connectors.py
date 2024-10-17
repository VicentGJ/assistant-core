from abc import ABC, abstractmethod
from io import BytesIO
from mimetypes import guess_type
from threading import local

from langchain_core.documents import Document
from pypdf import PdfReader
from storage3._sync.file_api import SyncBucket
from storage3.utils import StorageException
from supabase import Client, create_client

from modules.db.managers import DatabaseManager
from modules.types_models import SupabaseStorageFileData
from modules.utils.lib_utils import convert_date_string_to_datetime
from modules.vectorizers import VectorizerInterface
from settings import settings

thread_local = local()


class BaseConnector(ABC):
    def __init__(self, username: str) -> None:
        self.username = username

    @abstractmethod
    def validate_credentials(self) -> bool:
        pass

    @abstractmethod
    def get_files(self, db_manager: DatabaseManager) -> list:
        pass

    @abstractmethod
    def process_file(
        self,
        file,
        vectorstore: VectorizerInterface,
        db_manager: DatabaseManager,
        contextualize_docs: bool = False,
    ) -> str:
        pass

    @staticmethod
    def file_in_db(file_id, last_modified, db_manager: DatabaseManager) -> bool:
        db_doc = db_manager.get_document(file_id)
        if not db_doc:
            return False
        else:
            if convert_date_string_to_datetime(last_modified) > db_doc.last_modified:
                return False
            else:
                return True


class SupabaseStorageConnector(BaseConnector):
    def __init__(
        self,
        bucket_name: str = settings.supabase_storage_bucket_name,
        url: str = settings.supabase_url,
        key: str = settings.supabase_key,
        service_key: str = settings.supabase_service_key,
    ):
        self.supabase: Client = create_client(url, key)
        self.supabase.options.headers["Authorization"] = f"Bearer {service_key}"
        self.username = bucket_name
        bucket = self._get_bucket(bucket_name)
        if not bucket:
            raise ValueError("Bucket name not found")
        self.bucket = bucket
        self.storage_allowed_mime_types = [
            # "text/plain",
            # "application/msword",
            # "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            # "application/rtf",
            "application/pdf",
            # "application/vnd.oasis.opendocument.text",
            # "text/csv",
            # "text/markdown",
        ]
        self.url = url

    def validate_credentials(self) -> bool:
        try:
            buckets = self.supabase.storage.list_buckets()
            if self.username in [bucket.name for bucket in buckets]:
                return True
            else:
                return False
        except StorageException as e:
            error: dict = e.args[0]
            if error and error["message"] == "Unauthorized":
                return False
            else:
                raise StorageException(e.args)
        except Exception as e:
            raise StorageException(
                f"Failed validating credentials: An unexpected error {e.__class__.__name__} occurred: {str(e)}"
            )

    def _get_bucket(self, bucket_id: str) -> SyncBucket | None:
        try:
            return self.supabase.storage.get_bucket(bucket_id)
        except StorageException as e:
            error: dict = e.args[0]

            if error and error["message"] == "Bucket not found":
                return None
            else:
                raise StorageException(e.args)

    def _get_file(self, filename: str) -> bytes:
        try:
            print(filename)
            file_bytes = self.supabase.storage.from_(self.username).download(f"/{filename}")
            if type(file_bytes) is not bytes:
                raise StorageException(
                    "Failed downloading file from Supabase Storage. Response from Supabase Storage was not the file bytes."
                )
            return file_bytes
        except StorageException as e:
            raise StorageException(f"Failed downloading file from Supabase Storage: {str(e)}")
        except Exception as e:
            raise StorageException(
                f"Failed downloading file from Supabase Storage: An unexpected error {e.__class__.__name__} occurred: {str(e)}"
            )

    def _load_file(self, file_data: SupabaseStorageFileData) -> list[Document]:
        try:
            file_bytes = self._get_file(file_data.name)
            file_bytes_io = BytesIO(file_bytes)
            if not file_bytes:
                raise StorageException(
                    "Failed loading file from Supabase Storage. Response from Supabase Storage was empty."
                )
            reader = PdfReader(file_bytes_io)
            documents = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    if not file_data.metadata:
                        mime_type, _ = guess_type(file_data.name)
                    else:
                        mime_type = file_data.metadata.mimetype

                    # Create a Document for each page
                    doc = Document(
                        page_content=text,
                        metadata={
                            "id": file_data.id,
                            "name": file_data.name,
                            "updated_at": file_data.updated_at,
                            "provider": self.url,
                            "mime_type": mime_type,
                            "page": i + 1,
                        },
                    )
                    documents.append(doc)

            return documents
        except StorageException as e:
            raise StorageException(f"Failed loading file from Supabase Storage: {str(e)}")
        except Exception as e:
            raise StorageException(
                f"Failed loading file from Supabase Storage: An unexpected error {e.__class__.__name__} occurred: {str(e)}"
            )

    def get_files(self, db_manager: DatabaseManager) -> list[SupabaseStorageFileData]:
        try:
            file_elements: list[SupabaseStorageFileData] = []
            files_data = self.bucket.list()

            for file_data in files_data:
                if (
                    file_data["id"]
                    and not self.file_in_db(
                        file_id=file_data["id"],
                        last_modified=file_data["updated_at"],
                        db_manager=db_manager,
                    )
                    and file_data["metadata"]["mimetype"] in self.storage_allowed_mime_types
                ):
                    file_data = SupabaseStorageFileData(
                        name=file_data["name"],
                        id=file_data["id"],
                        updated_at=file_data["updated_at"],
                        created_at=file_data["created_at"],
                        last_accessed_at=file_data["last_accessed_at"],
                        metadata=file_data["metadata"],
                    )
                    file_elements.append(file_data)

            return file_elements
        except Exception as e:
            raise StorageException(
                f"Failed getting files: An unexpected error {e.__class__.__name__} occurred: {str(e)}"
            )

    def process_file(
        self,
        file,
        vectorizer: VectorizerInterface,
        db_manager: DatabaseManager,
        contextualize_docs: bool = False,
    ) -> str:
        try:
            docs = self._load_file(file)
            file_props = docs[0].metadata

            vectorizer.send_docs_to_vectorstore(docs, contextualize_docs)
            db_manager.register_document(
                username=self.username,
                file_id=file_props["id"],
                name=file_props["name"],
                last_modified=file_props["updated_at"],
                provider=file_props["provider"],
                doc_type=file_props["mime_type"],
            )
            return str(file_props["name"])
        except Exception as e:
            raise Exception(f"Failed processing file: An unexpected error {e.__class__.__name__} occurred: {str(e)}")
