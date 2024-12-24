from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your_secret_api_key"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")