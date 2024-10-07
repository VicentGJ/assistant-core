from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from modules.connectors import BaseConnector, SupabaseStorageConnector


security = HTTPBasic()


def check_connector_credentials(
    storage_bucket_name: str, credentials: HTTPBasicCredentials = Depends(security)
) -> BaseConnector:
    print(storage_bucket_name)
    print(credentials)
    connector: BaseConnector = SupabaseStorageConnector(bucket_name=storage_bucket_name)

    if not connector.validate_credentials():
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized user {credentials.username}",
        )
    return connector
