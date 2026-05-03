
from backend.models import Base, engine, StatcastBatterMetrics, StatcastPitcherMetrics
import backend.models

def create():
    Base.metadata.create_all(bind=engine)
    print("✅ Statcast tables created manually")

if __name__ == "__main__":
    create()
