from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from lib.connectors import BaseConnector


security = HTTPBasic()


def check_connector_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
) -> BaseConnector:
    connector: BaseConnector = BaseConnector(credentials.username, credentials.password)

    if not connector.validate_credentials():
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized user {connector.username}.",
        )
    return connector
