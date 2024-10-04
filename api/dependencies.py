from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from modules.connectors import ConnectorInterface, SupabaseStorageConnector


security = HTTPBasic()


def check_connector_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
) -> ConnectorInterface:
    connector: ConnectorInterface = SupabaseStorageConnector()

    if not connector.validate_credentials():
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized user {credentials.username}",
        )
    return connector
