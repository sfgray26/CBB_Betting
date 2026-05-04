
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import asyncio

async def check():
    yahoo = YahooFantasyClient()
    adp_players = await asyncio.to_thread(
        yahoo.get_adp_and_injury_feed,
        pages=1,
        count_per_page=5,
    )
    for p in adp_players:
        print(f"Key: {p.get('player_key')}, Owned: {p.get('percent_owned')}")

if __name__ == "__main__":
    asyncio.run(check())
