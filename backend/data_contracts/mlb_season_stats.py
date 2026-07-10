"""
MLBSeasonStats -- BDL /mlb/v1/season_stats season-aggregate row.

Ground truth: live probe 2026-07-10 against season=2026 for player_ids 40
(Cristopher Sanchez, pitcher) and 164 (Nasim Nunez, hitter):
  - One row per player per season/season_type
  - Batting fields null for pure pitchers; pitching fields null for pure hitters
  - Rate stats (batting_avg, pitching_era, pitching_whip) are floats
  - pitching_ip is a float (e.g. 120.1) -- MLB innings notation, NOT a decimal
  - player is the shared nested MLBPlayer object (includes team)

All stat fields are Optional -- never assume the API returns complete rows.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from backend.data_contracts.mlb_player import MLBPlayer


class MLBSeasonStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    player: Optional[MLBPlayer] = None
    team_name: Optional[str] = None
    season: Optional[int] = None
    postseason: Optional[bool] = None
    season_type: Optional[str] = None   # "regular" observed

    # ------------------------------------------------------------------
    # Batting (null for pure pitchers)
    # ------------------------------------------------------------------
    batting_gp: Optional[int] = None
    batting_ab: Optional[int] = None
    batting_r: Optional[int] = None
    batting_h: Optional[int] = None
    batting_avg: Optional[float] = None
    batting_2b: Optional[int] = None
    batting_3b: Optional[int] = None
    batting_hr: Optional[int] = None
    batting_rbi: Optional[int] = None
    batting_tb: Optional[int] = None
    batting_bb: Optional[int] = None
    batting_so: Optional[int] = None
    batting_sb: Optional[int] = None
    batting_obp: Optional[float] = None
    batting_slg: Optional[float] = None
    batting_ops: Optional[float] = None
    batting_war: Optional[float] = None

    # ------------------------------------------------------------------
    # Pitching (null for pure hitters)
    # ------------------------------------------------------------------
    pitching_gp: Optional[int] = None
    pitching_gs: Optional[int] = None
    pitching_qs: Optional[int] = None
    pitching_w: Optional[int] = None
    pitching_l: Optional[int] = None
    pitching_era: Optional[float] = None
    pitching_sv: Optional[int] = None
    pitching_hld: Optional[int] = None
    pitching_ip: Optional[float] = None   # MLB innings notation (120.1 = 120⅓)
    pitching_h: Optional[int] = None
    pitching_er: Optional[int] = None
    pitching_hr: Optional[int] = None
    pitching_bb: Optional[int] = None
    pitching_whip: Optional[float] = None
    pitching_k: Optional[int] = None
    pitching_k_per_9: Optional[float] = None
    pitching_war: Optional[float] = None

    # ------------------------------------------------------------------
    # Fielding
    # ------------------------------------------------------------------
    fielding_gp: Optional[int] = None
    fielding_gs: Optional[int] = None
    fielding_fip: Optional[float] = None
    fielding_tc: Optional[int] = None
    fielding_po: Optional[int] = None
    fielding_a: Optional[int] = None
    fielding_e: Optional[int] = None
    fielding_fp: Optional[float] = None

    @property
    def bdl_player_id(self) -> Optional[int]:
        """BDL player.id natural key. None if player object absent."""
        if self.player is None:
            return None
        return self.player.id

    @property
    def is_pitcher_line(self) -> bool:
        return self.pitching_gp is not None

    @property
    def is_batter_line(self) -> bool:
        return self.batting_gp is not None
