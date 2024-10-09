from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from modules.connectors import BaseConnector, SupabaseStorageConnector
from settings import settings
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def validate_token(token: str = Depends(oauth2_scheme)):
    print("TOKEN: ", token)
    if not token:
        raise HTTPException(status_code=401, detail="Api key not provided.")
    if token != settings.secret_key:
        raise HTTPException(status_code=401, detail="Invalid api key")


def check_connector_credentials(storage_bucket_name: str) -> BaseConnector:
    connector: BaseConnector = SupabaseStorageConnector(bucket_name=storage_bucket_name)

    if not connector.validate_credentials():
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized user",
        )
    return connector
