"""
Resilient Yahoo Fantasy API client with circuit breaker, fallback, and cache.

Drop-in replacement for YahooClient that handles:
- Circuit breaker for cascading failures
- Metadata fallback when percent_owned is unavailable
- Position normalization for lineup mismatches
- Stale cache for graceful degradation
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
from backend.fantasy_baseball.circuit_breaker import CircuitBreaker, CircuitOpenError
from backend.fantasy_baseball.cache_manager import StaleCacheManager, CacheResult, NoDataAvailableError
from backend.fantasy_baseball.position_normalizer import (
    PositionNormalizer, 
    YahooRoster, 
    RosterSlot, 
    Player,
    ValidationResult,
    LineupValidationError
)

logger = logging.getLogger(__name__)


@dataclass
class WaiverResponse:
    """Standardized waiver wire response."""
    players: List[Dict]
    source: str  # "yahoo_api", "cache", "projection_estimate"
    fresh: bool
    errors: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []


@dataclass
class LineupResult:
    """Result of lineup setting operation."""
    success: bool
    changes: Optional[Dict] = None
    errors: List[str] = None
    warnings: List[str] = None
    retry_possible: bool = False
    suggested_action: Optional[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ResilientYahooClient(YahooFantasyClient):
    """
    Yahoo client with resilience patterns.
    
    Extends the base YahooFantasyClient with:
    - Circuit breaker to prevent cascading failures
    - Fallback to metadata-only when percent_owned fails
    - Position normalization to prevent lineup mismatches
    - Stale cache for availability during outages
    
    Usage:
        client = ResilientYahooClient()  # Same init as YahooFantasyClient
        
        # Waiver wire with automatic fallback
        result = await client.get_waiver_players("mlb.l.12345")
        if not result.fresh:
            logger.warning(f"Serving stale data: {result.errors}")
        
        # Lineup with validation
        result = await client.set_lineup_resilient(team_id, optimized_lineup)
        if not result.success:
            print(result.suggested_action)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize resilience components
        self.circuit = CircuitBreaker(
            name="yahoo_fantasy_api",
            failure_threshold=3,
            recovery_timeout=300,  # 5 minutes
            expected_exception=Exception,
        )
        
        self.cache = StaleCacheManager(
            cache_dir=os.getenv("YAHOO_CACHE_DIR", ".cache/fantasy"),
            max_age_hours=int(os.getenv("YAHOO_CACHE_TTL_HOURS", "24")),
            enabled=os.getenv("YAHOO_CACHE_DISABLED", "false").lower() != "true"
        )
        
        self.position_normalizer = PositionNormalizer()
        
        # Track ADP data path for fallback enrichment
        self.adp_data_path = os.getenv(
            "ADP_DATA_PATH", 
            "/app/data/projections/adp_yahoo_2026.csv"
        )
    
    # ==================================================================
    # Waiver Wire Operations
    # ==================================================================
    
    async def get_waiver_players(self, league_id: str, **filters) -> WaiverResponse:
        """
        Get waiver wire players with full fallback chain.
        
        Fallback order:
        1. Try API with percent_owned (normal)
        2. Try API with metadata only + ADP enrichment
        3. Serve from cache if API unavailable
        4. Fail with clear error if no data available
        """
        cache_key = f"waiver_{league_id}_{hash(str(sorted(filters.items())))}"
        
        try:
            # Attempt 1: Circuit breaker wrapped API call with fallback
            players = await self.circuit.call_async(
                self._fetch_waiver_with_fallback,
                league_id,
                filters
            )
            
            # Cache successful result
            self.cache.write(cache_key, players, metadata={"filters": filters})
            
            return WaiverResponse(
                players=players,
                source="yahoo_api",
                fresh=True,
                errors=[],
                metadata={"count": len(players), "cached": False}
            )
            
        except CircuitOpenError:
            # Circuit open - use cache
            logger.warning("Circuit open for waiver fetch, checking cache")
            cached = self.cache.read(cache_key)
            
            if cached:
                return WaiverResponse(
                    players=cached.data,
                    source="cache",
                    fresh=False,
                    errors=["Yahoo API circuit open: serving stale data"],
                    metadata={
                        "count": len(cached.data),
                        "cached": True,
                        "cache_age_hours": self.cache.get_age_hours(cached)
                    }
                )
            
            # No cache available
            return WaiverResponse(
                players=[],
                source="unavailable",
                fresh=False,
                errors=["Yahoo API unavailable and no cache available"],
                metadata={"circuit_open": True}
            )
            
        except NoDataAvailableError as e:
            return WaiverResponse(
                players=[],
                source="unavailable",
                fresh=False,
                errors=[str(e)],
                metadata={"error_type": "no_data_available"}
            )
    
    async def _fetch_waiver_with_fallback(
        self, 
        league_id: str, 
        filters: Dict
    ) -> List[Dict]:
        """
        Internal: Try primary fetch, fallback to metadata-only on percent_owned error.
        """
        try:
            # Primary: Try normal fetch (assumes parent has this method)
            return await self._fetch_waiver_primary(league_id, filters)
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's the percent_owned error
            if "percent_owned" in error_str or "subresource" in error_str:
                logger.warning(
                    f"percent_owned subresource failed, using metadata fallback: {e}"
                )
                return await self._fetch_waiver_metadata_only(league_id, filters)
            
            # Re-raise other errors
            raise
    
    async def _fetch_waiver_primary(self, league_id: str, filters: Dict) -> List[Dict]:
        """Primary waiver fetch - override if parent method differs."""
        # Call parent implementation - adjust method name as needed
        if hasattr(super(), 'get_waiver_wire_players'):
            return await super().get_waiver_wire_players(league_id, **filters)
        elif hasattr(super(), 'get_waiver_players'):
            return await super().get_waiver_players(league_id, **filters)
        else:
            raise NotImplementedError("YahooClient waiver method not found")
    
    async def _fetch_waiver_metadata_only(
        self, 
        league_id: str, 
        filters: Dict
    ) -> List[Dict]:
        """
        Fallback: Fetch metadata only and enrich with ADP estimates.
        """
        logger.info(f"Fetching waiver metadata only for {league_id}")
        
        # Fetch without percent_owned subresource
        # This assumes YahooClient has a way to specify subresources
        # Adjust the call based on your actual YahooClient API
        
        players = await self._fetch_with_subresources(
            league_id, 
            subresources="metadata",
            **filters
        )
        
        # Enrich with ADP-based ownership estimates
        adp_data = self._load_adp_data()
        
        for player in players:
            player_name = player.get("name", "")
            estimated = self._estimate_ownership_from_adp(player_name, adp_data)
            player["percent_owned"] = estimated
            player["percent_owned_estimated"] = True
            player["percent_owned_source"] = "adp_proxy"
        
        return players
    
    async def _fetch_with_subresources(
        self, 
        league_id: str, 
        subresources: str,
        **filters
    ) -> List[Dict]:
        """Fetch players with specified subresources."""
        # This is a placeholder - implement based on your YahooClient API
        # You may need to override the URL construction to exclude percent_owned
        
        url = f"/fantasy/v2/league/{league_id}/players"
        params = {
            "out": subresources,
            "format": "json",
            **filters
        }
        
        # Call parent's request method
        response = await self._make_request(url, params)
        return self._parse_players_response(response)
    
    def _load_adp_data(self) -> Dict[str, float]:
        """Load ADP data for ownership estimation."""
        import csv
        
        adp_map = {}
        try:
            with open(self.adp_data_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "").strip()
                    adp = row.get("ADP", "")
                    if name and adp:
                        try:
                            adp_map[name] = float(adp)
                        except ValueError:
                            pass
        except FileNotFoundError:
            logger.warning(f"ADP data not found at {self.adp_data_path}")
        
        return adp_map
    
    def _estimate_ownership_from_adp(
        self, 
        player_name: str, 
        adp_data: Dict[str, float]
    ) -> float:
        """
        Estimate ownership percentage from ADP.
        
        Lower ADP (drafted earlier) = higher ownership
        This is a rough heuristic - adjust formula as needed.
        """
        adp = adp_data.get(player_name)
        if not adp:
            return 0.0  # Unknown players = 0% owned
        
        # Rough estimation: ADP 1-50 ~ 90-100%, ADP 200+ ~ 0-10%
        if adp <= 50:
            return max(0, 100 - (adp - 1) * 0.2)  # 100% at 1, 90% at 50
        elif adp <= 100:
            return max(0, 90 - (adp - 50) * 1.2)  # 90% at 50, 30% at 100
        elif adp <= 200:
            return max(0, 30 - (adp - 100) * 0.2)  # 30% at 100, 10% at 200
        else:
            return max(0, 10 - (adp - 200) * 0.05)  # 10% at 200, 0% at 400
    
    # ==================================================================
    # Lineup Operations
    # ==================================================================
    
    async def set_lineup_resilient(
        self, 
        team_id: str, 
        optimized_lineup: Dict[str, Any]
    ) -> LineupResult:
        """
        Set lineup with pre-validation and graceful degradation.
        
        Steps:
        1. Get current Yahoo roster
        2. Normalize positions between optimizer and Yahoo
        3. Validate before hitting API
        4. Execute with circuit breaker
        """
        try:
            # Step 1: Get current Yahoo roster
            yahoo_roster = await self._get_yahoo_roster(team_id)
            
            # Step 2: Normalize positions
            try:
                normalized_assignments = self.position_normalizer.normalize_lineup(
                    optimized_lineup, 
                    yahoo_roster,
                    strict=False  # Don't fail on unmatched slots
                )
            except LineupValidationError as e:
                return LineupResult(
                    success=False,
                    errors=[str(e)],
                    retry_possible=False,
                    suggested_action="Check position eligibility in optimizer vs Yahoo roster"
                )
            
            # Step 3: Validate before API call
            validation = self.position_normalizer.validate_lineup_before_submit(
                normalized_assignments,
                yahoo_roster
            )
            
            if not validation.valid:
                return LineupResult(
                    success=False,
                    errors=validation.errors,
                    warnings=validation.warnings,
                    retry_possible=False,
                    suggested_action="Fix position mismatches before retrying"
                )
            
            # Log warnings but proceed
            if validation.warnings:
                logger.warning(f"Lineup warnings: {validation.warnings}")
            
            # Step 4: Execute with circuit breaker
            try:
                result = await self.circuit.call_async(
                    self._execute_lineup_set,
                    team_id,
                    normalized_assignments
                )
                
                return LineupResult(
                    success=True,
                    changes=result,
                    warnings=validation.warnings,
                )
                
            except CircuitOpenError:
                return LineupResult(
                    success=False,
                    errors=["Yahoo API circuit is open (too many failures)"],
                    warnings=validation.warnings,
                    retry_possible=True,
                    suggested_action="Wait 5 minutes for circuit to reset, then retry"
                )
                
        except Exception as e:
            logger.exception("Unexpected error setting lineup")
            return LineupResult(
                success=False,
                errors=[f"Unexpected error: {str(e)}"],
                retry_possible=True,
                suggested_action="Check logs and retry"
            )
    
    async def _get_yahoo_roster(self, team_id: str) -> YahooRoster:
        """Fetch and parse Yahoo roster."""
        # This assumes parent has get_roster or similar
        roster_data = await self.get_roster(team_id)
        
        # Parse into YahooRoster structure
        slots = []
        players = []
        
        for slot_data in roster_data.get("slots", []):
            slots.append(RosterSlot(
                id=str(slot_data.get("slot_id") or slot_data.get("id")),
                position=slot_data.get("position", ""),
                player_id=str(slot_data.get("player_id")) if slot_data.get("player_id") else None
            ))
        
        for player_data in roster_data.get("players", []):
            players.append(Player(
                id=str(player_data.get("player_id") or player_data.get("id")),
                name=player_data.get("name", "Unknown"),
                positions=player_data.get("positions", []),
                yahoo_positions=player_data.get("eligible_positions", []),
                eligible_positions=player_data.get("eligible_positions", [])
            ))
        
        return YahooRoster(slots=slots, players=players)
    
    async def _execute_lineup_set(
        self, 
        team_id: str, 
        assignments: Dict[str, str]
    ) -> Dict:
        """Execute the actual lineup API call."""
        # Call parent implementation
        if hasattr(super(), 'set_lineup'):
            return await super().set_lineup(team_id, assignments)
        else:
            raise NotImplementedError("YahooClient set_lineup method not found")
    
    # ==================================================================
    # Health & Monitoring
    # ==================================================================
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all resilience components."""
        return {
            "circuit_breaker": self.circuit.get_stats(),
            "cache": self.cache.get_stats(),
            "client_type": "ResilientYahooClient",
        }
    
    def force_circuit_open(self):
        """Manually open circuit (for testing or emergency)."""
        self.circuit.force_open()
        logger.warning("Yahoo API circuit manually opened")
    
    def force_circuit_close(self):
        """Manually close circuit (after fixing issue)."""
        self.circuit.force_close()
        logger.info("Yahoo API circuit manually closed")
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear_all()
