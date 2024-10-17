from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file=".env")
    ENVIRONMENT: Literal["local", "dev", "qa", "prod"]
    db_url: str
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    openai_api_key: str
    cohere_api_key: str
    faiss_folder_path: str
    supabase_storage_bucket_name: str
    secret_key: str
    plataformia_api_key: str
    plataformia_base_url: str


settings = Settings()  # type: ignore
