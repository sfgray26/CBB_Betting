
from sqlalchemy import text
from backend.models import SessionLocal

def check():
    db = SessionLocal()
    try:
        res = db.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
        print([r[0] for r in res])
    finally:
        db.close()

if __name__ == "__main__":
    check()
