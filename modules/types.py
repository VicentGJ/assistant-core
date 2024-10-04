from pydantic import BaseModel


class SupabaseStorageFileMetadata(BaseModel):
    eTag: str
    size: int
    mimetype: str
    cacheControl: str
    lastModified: str
    contentLength: int
    httpStatusCode: int


class SupabaseStorageFileData(BaseModel):
    name: str
    id: str | None
    updated_at: str | None
    created_at: str | None
    last_accessed_at: str | None
    metadata: SupabaseStorageFileMetadata | None
