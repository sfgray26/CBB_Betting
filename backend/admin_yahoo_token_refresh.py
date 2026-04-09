"""
Admin endpoint to refresh Yahoo access token and return full token for Railway update.
TEMPORARY - Remove after token is updated in Railway.
"""

from fastapi import APIRouter
import requests
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/refresh-yahoo-token")
async def refresh_yahoo_token():
    """
    Refresh Yahoo access token using refresh token.
    Returns full access token for Railway environment variable update.

    REMOVE AFTER YAHOO_ACCESS_TOKEN IS SET IN RAILWAY!
    """
    try:
        client_id = os.getenv('YAHOO_CLIENT_ID')
        client_secret = os.getenv('YAHOO_CLIENT_SECRET')
        refresh_token = os.getenv('YAHOO_REFRESH_TOKEN')

        logger.info("Refreshing Yahoo access token...")

        response = requests.post('https://api.login.yahoo.com/oauth2/get_token', data={
            'client_id': client_id,
            'client_secret': client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }, timeout=10)

        if response.status_code == 200:
            tokens = response.json()
            access_token = tokens.get('access_token')
            expires_in = tokens.get('expires_in', 3600)

            logger.info("Token refresh successful")

            return {
                "status": "success",
                "access_token": access_token,
                "expires_in_seconds": expires_in,
                "railway_update_command": f'railway variables set YAHOO_ACCESS_TOKEN="{access_token}"',
                "message": "Copy the access_token and run the railway_update_command"
            }
        else:
            logger.error(f"Token refresh failed: {response.status_code}")
            return {
                "status": "error",
                "error": f"Refresh failed: {response.status_code} - {response.text[:200]}"
            }

    except Exception as e:
        logger.exception("Token refresh failed")
        return {"status": "error", "error": str(e)}
