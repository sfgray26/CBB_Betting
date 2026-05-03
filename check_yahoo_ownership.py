
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import asyncio
import json

async def check():
    yahoo = YahooFantasyClient()
    # Try with out=ownership
    data = await asyncio.to_thread(
        yahoo._get,
        f"league/{yahoo.league_key}/players;start=0;count=1;sort=DA;out=ownership"
    )
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(check())
