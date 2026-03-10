"""
Tests for conference_hca.py — P2 Conference-Specific HCA
"""

import pytest

from backend.services.conference_hca import (
    get_conference_hca,
    apply_conference_hca,
    normalize_conference_name,
    get_conference_difficulty_rating,
    CONFERENCE_HCA,
    DEFAULT_HCA,
)


class TestConferenceNormalization:
    """Test conference name normalization."""
    
    def test_big_ten_variations(self):
        """Should normalize various Big Ten names."""
        variations = ["Big Ten", "BIG TEN", "big ten", "Big 10", "B1G", "bigten"]
        for v in variations:
            assert normalize_conference_name(v) == "big_ten"
    
    def test_big_12_variations(self):
        """Should normalize various Big 12 names."""
        variations = ["Big 12", "BIG 12", "big 12", "Big12"]
        for v in variations:
            assert normalize_conference_name(v) == "big_12"
    
    def test_sec_variations(self):
        """Should normalize SEC names."""
        variations = ["SEC", "sec", "Southeastern Conference", "Southeastern"]
        for v in variations:
            assert normalize_conference_name(v) == "sec"
    
    def test_none_conference(self):
        """Should handle None gracefully."""
        assert normalize_conference_name(None) == "mid_major"
    
    def test_empty_conference(self):
        """Should handle empty string gracefully."""
        assert normalize_conference_name("") == "mid_major"
    
    def test_unknown_conference(self):
        """Should pass through unknown conferences."""
        assert normalize_conference_name("Some Random Conference") == "some_random_conference"


class TestConferenceHCAValues:
    """Test conference HCA retrieval."""
    
    def test_power_conference_values(self):
        """Power conferences should have higher HCA."""
        assert get_conference_hca("Big Ten") == 3.6
        assert get_conference_hca("Big 12") == 3.4
        assert get_conference_hca("SEC") == 3.2
        assert get_conference_hca("ACC") == 3.0
    
    def test_mid_major_values(self):
        """Mid-majors should have moderate HCA."""
        assert get_conference_hca("WCC") == 2.7
        assert get_conference_hca("AAC") == 2.6
        assert get_conference_hca("A-10") == 2.5
    
    def test_low_major_values(self):
        """Low-majors should have lower HCA."""
        assert get_conference_hca("SWAC") == 1.5
        assert get_conference_hca("MEAC") == 1.5
    
    def test_neutral_site_zero(self):
        """Neutral site should always be 0."""
        for conf in ["Big Ten", "SEC", "SWAC", None]:
            assert get_conference_hca(conf, is_neutral=True) == 0.0
    
    def test_unknown_conference_default(self):
        """Unknown conferences should get default HCA."""
        assert get_conference_hca("Unknown Conference") == DEFAULT_HCA


class TestApplyConferenceHCA:
    """Test HCA application with pace adjustment."""
    
    def test_basic_application(self):
        """Should apply conference HCA without pace adjustment."""
        hca, meta = apply_conference_hca("Big Ten", is_neutral=False, pace_ratio=1.0)
        
        assert hca == 3.6
        assert meta["conference"] == "Big Ten"
        assert meta["base_hca"] == 3.6
        assert meta["pace_ratio"] == 1.0
        assert meta["is_neutral"] is False
    
    def test_pace_adjustment(self):
        """Should scale HCA by pace ratio."""
        hca, meta = apply_conference_hca("Big Ten", is_neutral=False, pace_ratio=1.1)
        
        assert hca == pytest.approx(3.96, 0.01)  # 3.6 * 1.1
        assert meta["adjusted_hca"] == pytest.approx(3.96, 0.01)
    
    def test_neutral_site_override(self):
        """Should override conference HCA for neutral sites."""
        hca, meta = apply_conference_hca("Big Ten", is_neutral=True, pace_ratio=1.0)
        
        assert hca == 0.0
        assert meta["is_neutral"] is True
        assert meta["base_hca"] == 0.0
    
    def test_fast_pace_boost(self):
        """Fast pace should amplify HCA."""
        hca, _ = apply_conference_hca("Big Ten", pace_ratio=1.15)  # Fast game
        assert hca > 3.6
    
    def test_slow_pace_reduction(self):
        """Slow pace should reduce HCA."""
        hca, _ = apply_conference_hca("Big Ten", pace_ratio=0.85)  # Slow game
        assert hca < 3.6


class TestDifficultyRatings:
    """Test conference difficulty ratings."""
    
    def test_extreme_difficulty(self):
        """Big Ten and Big 12 should be EXTREME."""
        assert get_conference_difficulty_rating("Big Ten") == "EXTREME"
        assert get_conference_difficulty_rating("Big 12") == "EXTREME"
    
    def test_high_difficulty(self):
        """SEC and ACC should be HIGH."""
        assert get_conference_difficulty_rating("SEC") == "HIGH"
        assert get_conference_difficulty_rating("ACC") == "HIGH"
    
    def test_moderate_difficulty(self):
        """Big East, WCC should be MODERATE."""
        assert get_conference_difficulty_rating("Big East") == "MODERATE"
        assert get_conference_difficulty_rating("WCC") == "MODERATE"
    
    def test_standard_difficulty(self):
        """Most mid-majors should be STANDARD."""
        assert get_conference_difficulty_rating("C-USA") == "STANDARD"
        assert get_conference_difficulty_rating("MAC") == "STANDARD"
    
    def test_low_difficulty(self):
        """SWAC, MEAC should be LOW."""
        assert get_conference_difficulty_rating("SWAC") == "LOW"
        assert get_conference_difficulty_rating("MEAC") == "LOW"


class TestConferenceHCAIntegration:
    """Integration tests for conference HCA."""
    
    def test_all_conferences_have_values(self):
        """All defined conferences should have positive HCA."""
        for conf, hca in CONFERENCE_HCA.items():
            assert hca > 0, f"{conf} should have positive HCA"
            assert hca < 5, f"{conf} HCA seems too high: {hca}"
    
    def test_power_conference_ordering(self):
        """Power conferences should have higher HCA than mid-majors."""
        big_ten = get_conference_hca("Big Ten")
        wcc = get_conference_hca("WCC")
        swac = get_conference_hca("SWAC")
        
        assert big_ten > wcc > swac
    
    def test_case_insensitive_lookup(self):
        """Conference lookup should be case insensitive."""
        variations = ["BIG TEN", "big ten", "Big Ten", "bIg TeN"]
        results = [get_conference_hca(v) for v in variations]
        
        assert all(r == 3.6 for r in results)
