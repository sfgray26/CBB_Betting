"""
Tests for OpenClaw Morning Brief Generator.

Run with: python -m pytest tests/test_openclaw_briefs.py -v
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from backend.services.openclaw_briefs import (
    MorningBriefGenerator,
    BriefDataCollector,
    SlateSummary,
    IntegrityStatus,
    SharpMoneyAlert,
    TournamentStatus,
    generate_and_send_morning_brief,
)


class TestMorningBriefGenerator:
    """Test the MorningBriefGenerator class."""
    
    def test_generate_brief_with_all_data(self):
        """Test brief generation with complete data."""
        generator = MorningBriefGenerator()
        
        slate = SlateSummary(
            total_games=8,
            bet_tier=2,
            consider_tier=1,
            pass_tier=5,
            high_stakes_count=1
        )
        
        integrity = IntegrityStatus(all_confirmed=True)
        
        sharp_alerts = [
            SharpMoneyAlert(
                game="Duke @ UNC",
                line_move="-2.5 → -3.5",
                pattern="steam",
                confidence=0.85,
                recommendation="Wait for line to stabilize"
            )
        ]
        
        embed = generator.generate_brief(
            slate=slate,
            integrity=integrity,
            sharp_alerts=sharp_alerts,
            escalation_count=0
        )
        
        assert "title" in embed
        assert "description" in embed
        assert "color" in embed
        assert embed["color"] > 0  # Should have a color
        
        # Check title format
        assert "Morning Brief" in embed["title"]
        
        # Check description contains slate info
        desc = embed["description"]
        assert "8 games" in desc or "8 game" in desc
        assert "2 BET" in desc or "2 Bet" in desc
    
    def test_generate_brief_with_integrity_issues(self):
        """Test brief generation shows integrity issues prominently."""
        generator = MorningBriefGenerator()
        
        slate = SlateSummary(total_games=4, bet_tier=1)
        
        integrity = IntegrityStatus(
            all_confirmed=False,
            caution_count=1,
            volatile_count=1,
            abort_count=0,
            issues=["Gonzaga: Key player questionable"]
        )
        
        embed = generator.generate_brief(
            slate=slate,
            integrity=integrity,
            escalation_count=0
        )
        
        desc = embed["description"]
        assert "Integrity" in desc or "CAUTION" in desc or "VOLATILE" in desc
    
    def test_generate_brief_with_escalations(self):
        """Test brief generation shows escalations."""
        generator = MorningBriefGenerator()
        
        embed = generator.generate_brief(
            slate=SlateSummary(),
            escalation_count=3
        )
        
        desc = embed["description"]
        assert "3" in desc and "high-stakes" in desc.lower()
    
    def test_tournament_countdown_before_first_four(self):
        """Test tournament countdown before First Four."""
        generator = MorningBriefGenerator()
        
        # Mock today to be March 17, 2026 (1 day before First Four)
        generator.today = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
        generator.date_str = "March 17, 2026"
        generator.weekday = "Tuesday"
        
        embed = generator.generate_brief(slate=SlateSummary())
        
        # Should mention First Four
        desc = embed["description"]
        assert "First Four" in desc or "Tournament" in desc
    
    def test_color_red_for_abort(self):
        """Test that abort condition results in red color."""
        generator = MorningBriefGenerator()
        
        integrity = IntegrityStatus(abort_count=1)
        
        embed = generator.generate_brief(
            slate=SlateSummary(),
            integrity=integrity
        )
        
        # Red color in Discord is 0xE74C3C = 15158332
        assert embed["color"] == 15158332


class TestBriefDataCollector:
    """Test the BriefDataCollector class."""
    
    @pytest.mark.asyncio
    async def test_collect_slate_summary_no_db(self):
        """Test slate summary returns empty data when DB fails."""
        collector = BriefDataCollector(db_session=None)
        
        slate = await collector.collect_slate_summary()
        
        assert isinstance(slate, SlateSummary)
        assert slate.total_games == 0  # Should return empty on error
    
    @pytest.mark.asyncio
    async def test_collect_integrity_status(self):
        """Test integrity status collection."""
        collector = BriefDataCollector()
        
        integrity = await collector.collect_integrity_status()
        
        assert isinstance(integrity, IntegrityStatus)
        # Should return confirmed by default
        assert integrity.all_confirmed is True
    
    @pytest.mark.asyncio
    async def test_collect_escalation_count(self):
        """Test escalation count collection."""
        collector = BriefDataCollector()
        
        count = await collector.collect_escalation_count()
        
        assert isinstance(count, int)
        assert count >= 0


class TestIntegration:
    """Integration tests (may require DB)."""
    
    @pytest.mark.asyncio
    @patch("backend.services.openclaw_briefs.send_to_channel")
    async def test_generate_and_send_morning_brief(self, mock_send):
        """Test end-to-end brief generation and sending."""
        mock_send.return_value = True
        
        success = await generate_and_send_morning_brief()
        
        # Should attempt to send even if no data
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        
        # Verify it was called with openclaw-briefs channel (positional arg)
        args, kwargs = call_args
        assert args[0] == "openclaw-briefs"  # First positional arg is channel_name
        
        # Verify embed was provided (keyword arg)
        assert "embed" in kwargs


class TestCLIMode:
    """Test CLI functionality."""
    
    def test_module_runs_as_script(self):
        """Test that module can be run directly."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "backend.services.openclaw_briefs", "--test"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should generate test output
        assert "test brief" in result.stdout.lower() or result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
