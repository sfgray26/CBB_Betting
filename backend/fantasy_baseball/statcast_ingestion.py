"""
Statcast Daily Ingestion Pipeline

Pulls daily MLB data from Baseball Savant and updates player projections
using Bayesian inference with shrinkage priors.

Usage:
    from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent
    
    agent = StatcastIngestionAgent()
    agent.run_daily_ingestion()  # Pull yesterday's games and update projections

Schedule:
    Runs daily at 6:00 AM ET via scheduler in main.py
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.models import SessionLocal, PlayerProjection, StatcastPerformance
from backend.fantasy_baseball.player_board import get_or_create_projection

logger = logging.getLogger(__name__)


@dataclass
class PlayerDailyPerformance:
    """Single day performance from Statcast."""
    player_id: str
    player_name: str
    team: str
    game_date: date
    
    # Offense
    pa: int  # Plate appearances
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    r: int
    rbi: int
    bb: int
    so: int
    hbp: int
    sb: int
    cs: int
    
    # Statcast Quality Metrics
    exit_velocity_avg: float
    launch_angle_avg: float
    hard_hit_pct: float  # 95+ mph exit velocity
    barrel_pct: float    # Ideal EV/LA combination
    xba: float           # Expected batting average
    xslg: float          # Expected slugging
    xwoba: float         # Expected weighted on-base average
    
    # Pitching (if applicable)
    ip: float = 0.0
    er: int = 0
    k_pit: int = 0
    bb_pit: int = 0
    pitches: int = 0
    
    @property
    def avg(self) -> float:
        return self.h / self.ab if self.ab > 0 else 0.0
    
    @property
    def obp(self) -> float:
        pa_no_sf = self.pa  # Simplified
        return (self.h + self.bb + self.hbp) / pa_no_sf if pa_no_sf > 0 else 0.0
    
    @property
    def slg(self) -> float:
        tb = self.h + self.doubles + (2 * self.triples) + (3 * self.hr)
        return tb / self.ab if self.ab > 0 else 0.0
    
    @property
    def ops(self) -> float:
        return self.obp + self.slg
    
    @property
    def woba(self) -> float:
        """Calculate wOBA from components."""
        # Simplified wOBA calculation
        # 2024 weights: BB/HBP .69, 1B .89, 2B 1.27, 3B 1.62, HR 2.11
        singles = self.h - self.doubles - self.triples - self.hr
        numerator = (0.69 * (self.bb + self.hbp) + 
                     0.89 * singles + 
                     1.27 * self.doubles + 
                     1.62 * self.triples + 
                     2.11 * self.hr)
        return numerator / self.pa if self.pa > 0 else 0.0


@dataclass
class UpdatedProjection:
    """Result of Bayesian update."""
    player_id: str
    player_name: str
    
    # Prior (Steamer/ZiPS)
    prior_woba: float
    prior_variance: float
    
    # Likelihood (recent performance)
    sample_woba: float
    sample_variance: float
    sample_size: int  # PA
    
    # Posterior
    posterior_woba: float
    posterior_variance: float
    shrinkage: float  # 1.0 = trust prior fully, 0.0 = trust data fully
    
    # Additional stats
    updated_avg: float
    updated_obp: float
    updated_slg: float
    updated_ops: float
    updated_xwoba: float
    
    # Quality indicators
    data_quality_score: float  # 0-1 based on sample size, recency
    confidence_interval_95: Tuple[float, float]


class DataQualityChecker:
    """
    Validates ingested Statcast data for completeness and accuracy.
    """
    
    def __init__(self):
        self.issues: List[Dict] = []
    
    def validate_daily_pull(self, df: pd.DataFrame, target_date: date) -> bool:
        """
        Validate a daily Statcast pull.
        
        Checks:
        - Minimum games (should be ~15 for full slate)
        - Minimum players (should be ~300+)
        - Data completeness (% nulls)
        - Date range correctness
        """
        self.issues = []
        is_valid = True
        
        # Check 1: Minimum rows
        if len(df) < 200:
            self.issues.append({
                'severity': 'ERROR',
                'type': 'INSUFFICIENT_DATA',
                'message': f'Only {len(df)} player records, expected 300+',
                'target_date': target_date.isoformat()
            })
            is_valid = False
        
        # Check 2: Date range
        if 'game_date' in df.columns:
            dates = pd.to_datetime(df['game_date']).dt.date
            unique_dates = dates.unique()
            
            if target_date not in unique_dates:
                self.issues.append({
                    'severity': 'ERROR',
                    'type': 'WRONG_DATE',
                    'message': f'Target date {target_date} not in data. Found: {list(unique_dates)}',
                    'target_date': target_date.isoformat()
                })
                is_valid = False
            
            # Warn if multiple dates
            if len(unique_dates) > 1:
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'MULTIPLE_DATES',
                    'message': f'Data contains {len(unique_dates)} dates: {list(unique_dates)}',
                    'target_date': target_date.isoformat()
                })
        
        # Check 3: Critical columns present
        required_cols = ['player_name', 'team', 'game_date', 'pa', 'xwoba']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            self.issues.append({
                'severity': 'ERROR',
                'type': 'MISSING_COLUMNS',
                'message': f'Missing required columns: {missing_cols}',
                'target_date': target_date.isoformat()
            })
            is_valid = False
        
        # Check 4: Null rate
        if len(df) > 0:
            null_rates = df.isnull().mean()
            high_null_cols = null_rates[null_rates > 0.5].index.tolist()
            if high_null_cols:
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'HIGH_NULL_RATE',
                    'message': f'Columns with >50% nulls: {high_null_cols}',
                    'target_date': target_date.isoformat()
                })
        
        # Check 5: Value ranges
        if 'pa' in df.columns:
            invalid_pa = df[df['pa'] < 0]['pa'].count()
            if invalid_pa > 0:
                self.issues.append({
                    'severity': 'ERROR',
                    'type': 'INVALID_DATA',
                    'message': f'{invalid_pa} rows with negative PA',
                    'target_date': target_date.isoformat()
                })
                is_valid = False
        
        # Check 6: Reasonable xwoba range (0.000 to 0.600 is typical)
        if 'xwoba' in df.columns:
            outlier_xwoba = df[(df['xwoba'] < 0.000) | (df['xwoba'] > 0.700)]['xwoba'].count()
            if outlier_xwoba > len(df) * 0.05:  # More than 5% outliers
                self.issues.append({
                    'severity': 'WARNING',
                    'type': 'DATA_ANOMALY',
                    'message': f'{outlier_xwoba} players with unusual xwoba values',
                    'target_date': target_date.isoformat()
                })
        
        return is_valid
    
    def get_validation_report(self) -> Dict:
        """Generate validation report for logging/monitoring."""
        errors = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        
        return {
            'is_valid': len(errors) == 0,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'errors': errors,
            'warnings': warnings
        }


class StatcastIngestionAgent:
    """
    Agent responsible for daily Statcast data ingestion.
    
    Orchestrates:
    1. Fetching yesterday's data from Baseball Savant
    2. Validating data quality
    3. Running Bayesian projection updates
    4. Storing results in database
    5. Logging issues for monitoring
    """
    
    def __init__(self):
        self.base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        self.quality_checker = DataQualityChecker()
        self.db = SessionLocal()
    
    def fetch_statcast_day(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch Statcast data for a specific date.
        
        Uses Baseball Savant CSV export API.
        """
        logger.info(f"Fetching Statcast data for {target_date}")
        
        # Build query parameters
        from datetime import timedelta
        params = {
            'all': 'true',
            'hfPT': '',  # Pitch types (all)
            'hfAB': '',  # Batted ball types (all)
            'hfBBT': '',  # Batted ball trajectory (all)
            'hfPR': '',  # Pitch result (all)
            'hfZ': '',  # Zone (all)
            'stadium': '',  # All stadiums
            'hfBBL': '',  # Barrel (all)
            'hfNewZones': '0',
            'hfGT': 'R%7C',  # Game type: Regular season
            'hfC': '',  # Count (all)
            'hfSea': f'{target_date.year}%7C',  # Season
            'hfSit': '',  # Situation (all)
            'player_type': 'batter',  # Batter perspective
            'hfOuts': '',  # Outs (all)
            'opponent': '',  # All opponents
            'pitcher_throws': '',  # L/R (all)
            'batter_stands': '',  # L/R (all)
            'hfSA': '',  # Sacrifice (all)
            'game_date_gt': (target_date - timedelta(days=1)).isoformat(),
            'game_date_lt': (target_date + timedelta(days=1)).isoformat(),
            'hfInfield': '',  # Infield (all)
            'team': '',  # All teams
            'position': '',  # All positions
            'hfOutfield': '',  # Outfield (all)
            'hfRO': '',  # Runner on (all)
            'home_road': '',  # Home/Road (all)
            'hfFlag': '',  # Flag (all)
            'metric_1': '',
            'group_by': 'name-date',  # Group by player and date
            'min_pitches': '0',
            'min_results': '0',
            'min_pas': '0',
            'sort_col': 'pitches',
            'player_event_sort': 'api_p_release_speed',
            'sort_order': 'desc',
            'type': 'details'
        }
        
        try:
            # Note: Baseball Savant requires proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Statcast API returned {response.status_code}")
                return None
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"Successfully fetched {len(df)} records from Statcast")
            return df
            
        except Exception as e:
            logger.exception(f"Failed to fetch Statcast data: {e}")
            return None
    
    def transform_to_performance(self, df: pd.DataFrame) -> List[PlayerDailyPerformance]:
        """
        Transform Statcast DataFrame to PlayerDailyPerformance objects.
        """
        performances = []
        
        for _, row in df.iterrows():
            try:
                perf = PlayerDailyPerformance(
                    player_id=str(row.get('player_id', '')),
                    player_name=str(row.get('player_name', '')),
                    team=str(row.get('team', '')),
                    game_date=pd.to_datetime(row.get('game_date')).date(),
                    
                    pa=int(row.get('pa', 0)),
                    ab=int(row.get('ab', 0)),
                    h=int(row.get('hit', 0)),
                    doubles=int(row.get('double', 0)),
                    triples=int(row.get('triple', 0)),
                    hr=int(row.get('home_run', 0)),
                    r=int(row.get('run', 0)),
                    rbi=int(row.get('rbi', 0)),
                    bb=int(row.get('walk', 0)),
                    so=int(row.get('strikeout', 0)),
                    hbp=int(row.get('hbp', 0)),
                    sb=int(row.get('sb', 0)),
                    cs=int(row.get('cs', 0)),
                    
                    exit_velocity_avg=float(row.get('exit_velocity_avg', 0) or 0),
                    launch_angle_avg=float(row.get('launch_angle_avg', 0) or 0),
                    hard_hit_pct=float(row.get('hard_hit_percent', 0) or 0) / 100,
                    barrel_pct=float(row.get('barrel_batted_rate', 0) or 0) / 100,
                    xba=float(row.get('xba', 0) or 0),
                    xslg=float(row.get('xslg', 0) or 0),
                    xwoba=float(row.get('xwoba', 0) or 0),
                    
                    ip=float(row.get('ip', 0) or 0),
                    er=int(row.get('er', 0) or 0),
                    k_pit=int(row.get('k', 0) or 0),
                    bb_pit=int(row.get('bb', 0) or 0),
                    pitches=int(row.get('pitches', 0) or 0)
                )
                performances.append(perf)
            except Exception as e:
                logger.warning(f"Failed to parse row for {row.get('player_name', 'unknown')}: {e}")
                continue
        
        return performances
    
    def store_performances(self, performances: List[PlayerDailyPerformance]):
        """Store daily performances in database."""
        for perf in performances:
            try:
                # Check if already exists
                existing = self.db.query(StatcastPerformance).filter(
                    StatcastPerformance.player_id == perf.player_id,
                    StatcastPerformance.game_date == perf.game_date
                ).first()
                
                if existing:
                    logger.debug(f"Performance already exists for {perf.player_name} on {perf.game_date}")
                    continue
                
                # Create new record
                record = StatcastPerformance(
                    player_id=perf.player_id,
                    player_name=perf.player_name,
                    team=perf.team,
                    game_date=perf.game_date,
                    
                    pa=perf.pa,
                    ab=perf.ab,
                    h=perf.h,
                    doubles=perf.doubles,
                    triples=perf.triples,
                    hr=perf.hr,
                    r=perf.r,
                    rbi=perf.rbi,
                    bb=perf.bb,
                    so=perf.so,
                    hbp=perf.hbp,
                    sb=perf.sb,
                    cs=perf.cs,
                    
                    exit_velocity_avg=perf.exit_velocity_avg,
                    launch_angle_avg=perf.launch_angle_avg,
                    hard_hit_pct=perf.hard_hit_pct,
                    barrel_pct=perf.barrel_pct,
                    xba=perf.xba,
                    xslg=perf.xslg,
                    xwoba=perf.xwoba,
                    
                    woba=perf.woba,
                    avg=perf.avg,
                    obp=perf.obp,
                    slg=perf.slg,
                    ops=perf.ops,
                    
                    created_at=datetime.now(ZoneInfo("America/New_York"))
                )
                
                self.db.add(record)
                
            except Exception as e:
                logger.warning(f"Failed to store performance for {perf.player_name}: {e}")
                continue
        
        self.db.commit()
        logger.info(f"Stored {len(performances)} daily performances")


class BayesianProjectionUpdater:
    """
    Updates player projections using Bayesian inference with shrinkage priors.
    
    Key insight: Early season data is noisy. Use shrinkage to balance prior
    (Steamer/ZiPS) with likelihood (actual performance).
    
    shrinkage = 1.0 → Trust prior fully (no season data)
    shrinkage = 0.0 → Trust data fully (large sample)
    """
    
    def __init__(self):
        self.db = SessionLocal()
    
    def get_prior_projection(self, player_id: str) -> Optional[Dict]:
        """Get Steamer/ZiPS prior projection for a player."""
        # Try database first
        db_proj = self.db.query(PlayerProjection).filter(
            PlayerProjection.player_id == player_id
        ).first()
        
        if db_proj:
            return {
                'player_id': player_id,
                'woba': db_proj.woba,
                'avg': db_proj.avg,
                'obp': db_proj.obp,
                'slg': db_proj.slg,
                'ops': db_proj.ops,
                'hr': db_proj.hr,
                'r': db_proj.r,
                'rbi': db_proj.rbi,
                'sb': db_proj.sb,
                'variance': 0.0025  # Assumed variance in prior
            }
        
        # Fall back to player_board
        yahoo_player = {'player_key': player_id}
        board_proj = get_or_create_projection(yahoo_player)
        
        if board_proj:
            return {
                'player_id': player_id,
                'woba': board_proj.get('woba', 0.320),
                'avg': board_proj.get('avg', 0.250),
                'obp': board_proj.get('obp', 0.320),
                'slg': board_proj.get('slg', 0.400),
                'ops': board_proj.get('ops', 0.720),
                'hr': board_proj.get('hr', 15),
                'r': board_proj.get('r', 65),
                'rbi': board_proj.get('rbi', 65),
                'sb': board_proj.get('sb', 5),
                'variance': 0.0025
            }
        
        return None
    
    def get_recent_performance(
        self, 
        player_id: str, 
        lookback_days: int = 14
    ) -> Optional[Dict]:
        """
        Get aggregated performance over last N days.
        
        Returns weighted performance (recent games weighted more heavily).
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        performances = self.db.query(StatcastPerformance).filter(
            StatcastPerformance.player_id == player_id,
            StatcastPerformance.game_date >= start_date,
            StatcastPerformance.game_date <= end_date
        ).all()
        
        if not performances:
            return None
        
        # Calculate weighted averages (exponential decay)
        total_pa = sum(p.pa for p in performances)
        if total_pa < 10:  # Need minimum sample
            return None
        
        # Simple unweighted for now (can add decay later)
        weighted_woba = sum(p.woba * p.pa for p in performances) / total_pa
        weighted_xwoba = sum(p.xwoba * p.pa for p in performances) / total_pa
        weighted_avg = sum(p.avg * p.pa for p in performances) / total_pa
        weighted_slg = sum(p.slg * p.pa for p in performances) / total_pa
        weighted_ops = sum(p.ops * p.pa for p in performances) / total_pa
        
        # Calculate sample variance (simplified)
        sample_variance = 0.01 / total_pa if total_pa > 0 else 1.0
        
        return {
            'woba': weighted_woba,
            'xwoba': weighted_xwoba,
            'avg': weighted_avg,
            'slg': weighted_slg,
            'ops': weighted_ops,
            'pa': total_pa,
            'games': len(performances),
            'variance': sample_variance
        }
    
    def bayesian_update(
        self,
        prior: Dict,
        likelihood: Dict
    ) -> UpdatedProjection:
        """
        Perform conjugate normal update.
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        """
        player_id = prior['player_id']
        
        # Prior parameters
        prior_mean = prior['woba']
        prior_precision = 1 / prior['variance']
        
        # Likelihood parameters
        sample_mean = likelihood['woba']
        sample_variance = likelihood['variance']
        likelihood_precision = 1 / sample_variance if sample_variance > 0 else 0
        
        # Posterior calculation
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (
            (prior_precision * prior_mean) + 
            (likelihood_precision * sample_mean)
        ) / posterior_precision
        
        # Shrinkage factor
        shrinkage = prior_precision / posterior_precision
        
        # Confidence interval (95%)
        posterior_std = (1 / posterior_precision) ** 0.5
        ci_lower = posterior_mean - (1.96 * posterior_std)
        ci_upper = posterior_mean + (1.96 * posterior_std)
        
        # Data quality score based on sample size
        # 0 PA = 0.0 quality, 200+ PA = 1.0 quality
        data_quality = min(1.0, likelihood['pa'] / 200)
        
        return UpdatedProjection(
            player_id=player_id,
            player_name=prior.get('player_name', player_id),
            prior_woba=prior_mean,
            prior_variance=prior['variance'],
            sample_woba=sample_mean,
            sample_variance=sample_variance,
            sample_size=likelihood['pa'],
            posterior_woba=posterior_mean,
            posterior_variance=1 / posterior_precision,
            shrinkage=shrinkage,
            updated_avg=likelihood['avg'],  # Simplified: use recent avg
            updated_obp=prior['obp'],  # Would calculate properly
            updated_slg=likelihood['slg'],
            updated_ops=likelihood['ops'],
            updated_xwoba=likelihood['xwoba'],
            data_quality_score=data_quality,
            confidence_interval_95=(ci_lower, ci_upper)
        )
    
    def update_all_projections(self, min_pa: int = 20) -> List[UpdatedProjection]:
        """
        Run Bayesian update for all players with sufficient recent data.
        
        Args:
            min_pa: Minimum plate appearances to trigger update
        """
        logger.info(f"Running Bayesian projection updates (min {min_pa} PA)")
        
        updated_projections = []
        
        # Get all players with recent data
        recent = self.db.query(StatcastPerformance.player_id).distinct().all()
        player_ids = [p[0] for p in recent]
        
        logger.info(f"Found {len(player_ids)} players with recent Statcast data")
        
        for player_id in player_ids:
            try:
                # Get prior
                prior = self.get_prior_projection(player_id)
                if not prior:
                    logger.debug(f"No prior projection for {player_id}")
                    continue
                
                # Get recent performance
                likelihood = self.get_recent_performance(player_id, lookback_days=14)
                if not likelihood or likelihood['pa'] < min_pa:
                    continue
                
                # Run Bayesian update
                updated = self.bayesian_update(prior, likelihood)
                updated_projections.append(updated)
                
                # Store in database
                self._store_updated_projection(updated)
                
            except Exception as e:
                logger.warning(f"Failed to update projection for {player_id}: {e}")
                continue
        
        self.db.commit()
        logger.info(f"Updated {len(updated_projections)} player projections")
        
        return updated_projections
    
    def _store_updated_projection(self, updated: UpdatedProjection):
        """Store updated projection in database."""
        # Check for existing
        existing = self.db.query(PlayerProjection).filter(
            PlayerProjection.player_id == updated.player_id
        ).first()
        
        if existing:
            # Update existing
            existing.woba = updated.posterior_woba
            existing.avg = updated.updated_avg
            existing.obp = updated.updated_obp
            existing.slg = updated.updated_slg
            existing.ops = updated.updated_ops
            existing.xwoba = updated.updated_xwoba
            existing.shrinkage = updated.shrinkage
            existing.data_quality_score = updated.data_quality_score
            existing.sample_size = updated.sample_size
            existing.updated_at = datetime.now(ZoneInfo("America/New_York"))
            existing.update_method = 'bayesian'
        else:
            # Create new
            record = PlayerProjection(
                player_id=updated.player_id,
                player_name=updated.player_name,
                woba=updated.posterior_woba,
                avg=updated.updated_avg,
                obp=updated.updated_obp,
                slg=updated.updated_slg,
                ops=updated.updated_ops,
                xwoba=updated.updated_xwoba,
                shrinkage=updated.shrinkage,
                data_quality_score=updated.data_quality_score,
                sample_size=updated.sample_size,
                updated_at=datetime.now(ZoneInfo("America/New_York")),
                update_method='bayesian'
            )
            self.db.add(record)


def run_daily_ingestion(target_date: Optional[date] = None):
    """
    Main entry point for daily Statcast ingestion.
    
    This function is called by:
    - Scheduled job (6:00 AM ET daily)
    - Manual trigger from admin panel
    - Backfill script for historical data
    
    Args:
        target_date: Date to ingest (default: yesterday)
    """
    if target_date is None:
        # Anchor to ET — Railway runs UTC so date.today() can be wrong after midnight ET
        target_date = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
    
    logger.info("=" * 60)
    logger.info(f"Starting daily Statcast ingestion for {target_date}")
    logger.info("=" * 60)
    
    # Step 1: Ingest data
    agent = StatcastIngestionAgent()
    df = agent.fetch_statcast_day(target_date)
    
    if df is None or len(df) == 0:
        logger.error(f"Failed to fetch Statcast data for {target_date}")
        return {
            'success': False,
            'date': target_date.isoformat(),
            'error': 'Failed to fetch data',
            'records_processed': 0
        }
    
    # Step 2: Validate data quality
    is_valid = agent.quality_checker.validate_daily_pull(df, target_date)
    validation_report = agent.quality_checker.get_validation_report()
    
    logger.info(f"Data validation: {validation_report['error_count']} errors, "
                f"{validation_report['warning_count']} warnings")
    
    if not is_valid:
        logger.error("Data quality validation failed")
        for error in validation_report['errors']:
            logger.error(f"  - {error['type']}: {error['message']}")
    
    # Step 3: Transform and store
    performances = agent.transform_to_performance(df)
    agent.store_performances(performances)
    
    # Step 4: Run Bayesian updates
    updater = BayesianProjectionUpdater()
    updated_projections = updater.update_all_projections(min_pa=20)
    
    # Step 5: Generate summary
    high_confidence = [p for p in updated_projections if p.data_quality_score > 0.5]
    big_movers = [p for p in updated_projections 
                  if abs(p.posterior_woba - p.prior_woba) > 0.020]
    
    logger.info("=" * 60)
    logger.info(f"Daily ingestion complete for {target_date}")
    logger.info(f"  Records processed: {len(performances)}")
    logger.info(f"  Projections updated: {len(updated_projections)}")
    logger.info(f"  High confidence (>50% quality): {len(high_confidence)}")
    logger.info(f"  Big movers (>20 wOBA points): {len(big_movers)}")
    logger.info("=" * 60)
    
    return {
        'success': True,
        'date': target_date.isoformat(),
        'records_processed': len(performances),
        'projections_updated': len(updated_projections),
        'high_confidence_updates': len(high_confidence),
        'big_movers': len(big_movers),
        'validation': validation_report,
        'big_mover_details': [
            {
                'name': p.player_name,
                'prior': round(p.prior_woba, 3),
                'posterior': round(p.posterior_woba, 3),
                'delta': round(p.posterior_woba - p.prior_woba, 3),
                'shrinkage': round(p.shrinkage, 3)
            }
            for p in big_movers[:10]  # Top 10
        ]
    }


if __name__ == "__main__":
    # Run for yesterday when called directly
    import sys
    
    if len(sys.argv) > 1:
        # Parse date from command line: YYYY-MM-DD
        target = date.fromisoformat(sys.argv[1])
    else:
        target = None
    
    result = run_daily_ingestion(target)
    print(result)
