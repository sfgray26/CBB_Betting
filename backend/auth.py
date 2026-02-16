"""
Simple API Key authentication for personal use
Much simpler than Azure Entra ID for 1-2 users
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os
from typing import Dict
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Valid API keys from environment
def get_valid_api_keys() -> Dict[str, str]:
    """Load valid API keys from environment variables"""
    keys = {}
    
    # Support up to 5 users
    for i in range(1, 6):
        key = os.getenv(f"API_KEY_USER{i}")
        if key:
            keys[key] = f"user{i}"
    
    if not keys:
        # Development fallback (never use in production)
        if os.getenv("ENVIRONMENT") == "development":
            keys["dev-key-insecure"] = "dev_user"
        else:
            raise ValueError("No API keys configured! Set API_KEY_USER1 in environment")
    
    return keys


VALID_API_KEYS = get_valid_api_keys()


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Verify API key and return user identifier
    
    Usage in FastAPI routes:
        @app.get("/protected")
        async def protected_route(user: str = Depends(verify_api_key)):
            return {"user": user}
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return VALID_API_KEYS[api_key]


async def verify_admin_api_key(user: str = Security(verify_api_key)) -> str:
    """
    Admin-only routes (only user1 is admin)
    
    Usage:
        @app.post("/admin/recalibrate")
        async def admin_route(user: str = Depends(verify_admin_api_key)):
            ...
    """
    if user != "user1":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user
