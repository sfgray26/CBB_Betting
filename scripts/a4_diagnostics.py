import os
import sqlalchemy
from sqlalchemy import text

def diagnose():
    url = os.environ.get('DATABASE_URL')
    if not url:
        print("Error: DATABASE_URL not set")
        return
    
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print('--- Decision Pipeline Diagnostics (A-4) ---')
        tables = ['mlb_game_log', 'mlb_player_stats', 'player_rolling_stats', 'player_scores', 'decision_results']
        for table in tables:
            try:
                count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
                latest = 'N/A'
                if table == 'mlb_game_log':
                    latest = conn.execute(text('SELECT MAX(game_date) FROM mlb_game_log')).scalar()
                elif table == 'mlb_player_stats':
                    latest = conn.execute(text('SELECT MAX(game_date) FROM mlb_player_stats')).scalar()
                elif table == 'player_rolling_stats':
                    latest = conn.execute(text('SELECT MAX(as_of_date) FROM player_rolling_stats')).scalar()
                elif table == 'player_scores':
                    latest = conn.execute(text('SELECT MAX(as_of_date) FROM player_scores')).scalar()
                elif table == 'decision_results':
                    latest = conn.execute(text('SELECT MAX(created_at) FROM decision_results')).scalar()
                print(f'{table:20} | Count: {count:7} | Latest: {latest}')
            except Exception as e:
                print(f'{table:20} | Error: {e}')

if __name__ == "__main__":
    diagnose()
