# -*- coding: utf-8 -*-
"""Regression tests for accent handling in identity name normalization.

Root cause (2026-07-10): _normalize_identity_name used NFKD decomposition but
never stripped the combining marks it produced, so Yahoo's accented names
("Cristopher Sánchez") failed equality checks against player_id_mapping's
unaccented normalized_name ("cristopher sanchez"). The resolver then rejected
CORRECT mapping rows and accented-name players fell to projection_fallback.
"""
import pytest

from backend.routers.fantasy import _mapping_name_matches, _normalize_identity_name


class TestNormalizeIdentityName:
    @pytest.mark.parametrize(
        "accented, expected",
        [
            ("Cristopher Sánchez", "cristopher sanchez"),
            ("Nasim Nuñez", "nasim nunez"),
            ("Edwin Díaz", "edwin diaz"),
            ("Ronald Acuña Jr.", "ronald acuna"),
            ("José Ramírez", "jose ramirez"),
        ],
    )
    def test_accents_are_stripped(self, accented, expected):
        assert _normalize_identity_name(accented) == expected

    def test_suffix_and_dots_still_removed(self):
        assert _normalize_identity_name("Luis Robert Jr.") == "luis robert"
        assert _normalize_identity_name("J.T. Realmuto") == "jt realmuto"

    def test_empty_and_none_safe(self):
        assert _normalize_identity_name("") == ""
        assert _normalize_identity_name(None) == ""


class TestMappingNameMatches:
    def test_accented_yahoo_name_matches_unaccented_mapping_row(self):
        # The exact production failure: Yahoo roster name vs DB normalized_name
        assert _mapping_name_matches("Cristopher Sánchez", "cristopher sanchez")
        assert _mapping_name_matches("Nasim Nuñez", "nasim nunez")

    def test_different_players_still_rejected(self):
        assert not _mapping_name_matches("Cristopher Sánchez", "juan soto")
