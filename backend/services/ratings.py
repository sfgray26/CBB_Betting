"""
CBB Ratings data from multiple sources
- KenPom (Official API with Bearer Token)
- BartTorvik (Scraping/CSV)
- EvanMiya (Scraping)
"""

import requests
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from backend.services.team_mapping import normalize_team_name

load_dotenv()

logger = logging.getLogger(__name__)


class RatingsService:
    """Multi-source ratings aggregator with automated matching"""
    
    def __init__(self):
        self.kenpom_key = os.getenv("KENPOM_API_KEY")
        self.cache = {}
        self.cache_timestamp = None

    def get_team_rating(self, team_name: str, source_data: Dict[str, float]) -> Optional[float]:
        """Safely extract a rating using the centralized normalization service."""
        if not source_data:
            return None
        
        # Use the new, centralized normalization function
        normalized = normalize_team_name(team_name, list(source_data.keys()))
        
        if normalized is None:
            logger.warning(f"❌ Mismatch: '{team_name}' could not be normalized or found in source.")
            return None

        rating = source_data.get(normalized)
        
        if rating is None:
            # This case should be rare if normalization is good, but good to have
            logger.warning(f"❌ Mismatch: '{team_name}' (normalized: '{normalized}') not found in source post-normalization.")
        
        return rating
    
    def get_kenpom_ratings(self) -> Dict[str, float]:
        if not self.kenpom_key:
            logger.warning("⚠️ KenPom API key not set")
            return {}
        
        url = "https://kenpom.com/api.php"
        headers = {"Authorization": f"Bearer {self.kenpom_key}", "Accept": "application/json"}
        params = {"endpoint": "ratings", "y": 2026}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            ratings = {}
            for t in data:
                name = t.get("TeamName")
                adj_em = t.get("AdjEM")
                if name and adj_em is not None:
                    try:
                        ratings[name.strip()] = float(adj_em)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse AdjEM for {name}: {adj_em!r}")
            logger.info(f"✅ KenPom: Loaded {len(ratings)} teams")
            return ratings
        except Exception as e:
            logger.error(f"❌ KenPom API error: {e}")
            return {}

    def get_barttorvik_ratings(self) -> Dict[str, float]:
        url = "https://barttorvik.com/2026_team_results.csv"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            ratings = {}
            lines = response.text.strip().split('\n')
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) > 5:
                    ratings[parts[1].strip()] = float(parts[5])
            logger.info(f"✅ BartTorvik: Loaded {len(ratings)} teams")
            return ratings
        except Exception as e:
            logger.warning(f"⚠️ BartTorvik CSV error: {e}")
            return {}

    def get_evanmiya_ratings(self) -> Dict[str, float]:
        return {} # Placeholder

    def get_all_ratings(self, use_cache: bool = True) -> Dict[str, Dict[str, float]]:
        if use_cache and self.cache and self.cache_timestamp:
            age = (datetime.utcnow() - self.cache_timestamp).total_seconds() / 3600
            if age < 6: return self.cache
        
        ratings = {
            'kenpom': self.get_kenpom_ratings(),
            'barttorvik': self.get_barttorvik_ratings(),
            'evanmiya': self.get_evanmiya_ratings()
        }
        self.cache = ratings
        self.cache_timestamp = datetime.utcnow()
        return ratings

# Singleton
_ratings_service = None
def get_ratings_service() -> RatingsService:
    global _ratings_service
    if _ratings_service is None:
        _ratings_service = RatingsService()
    return _ratings_service

def fetch_current_ratings() -> Dict:
    return get_ratings_service().get_all_ratings()