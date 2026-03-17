"""
Tests for generate_pool_optimal_bracket() — pool-optimal bracket generator.

Run: pytest tests/test_pool_optimal_bracket.py -v
"""

import pytest
from backend.tournament.matchup_predictor import TournamentTeam
from backend.tournament.smart_bracket import generate_pool_optimal_bracket


# ---------------------------------------------------------------------------
# Fixture: minimal 4-region bracket (4 teams per region for speed)
# We use seeds 1,5,8,12 per region to hit the key matchup types.
# ---------------------------------------------------------------------------

def _team(name, seed, region, rating):
    return TournamentTeam(
        name=name, seed=seed, region=region,
        composite_rating=rating,
        kp_adj_em=None, bt_adj_em=None,
    )


def _make_bracket():
    """Build a minimal 16-team bracket (4 per region) covering 8v9 and 5v12 matchups."""
    regions = {
        "east": [
            _team("Duke",          1,  "east",  28.5),
            _team("UConn",         2,  "east",  22.5),
            _team("Michigan State",3,  "east",  19.5),
            _team("Kansas",        4,  "east",  16.5),
            _team("St. Johns",     5,  "east",  13.5),
            _team("Louisville",    6,  "east",  11.5),
            _team("UCLA",          7,  "east",   9.5),
            _team("Ohio State",    8,  "east",   7.5),
            _team("TCU",           9,  "east",   5.5),
            _team("UCF",           10, "east",   3.5),
            _team("South Florida", 11, "east",   1.5),
            _team("Northern Iowa", 12, "east",  -0.5),
            _team("Cal Baptist",   13, "east",  -2.5),
            _team("N Dakota St",   14, "east",  -4.5),
            _team("Furman",        15, "east",  -6.5),
            _team("Siena",         16, "east",  -9.0),
        ],
        "south": [
            _team("Florida",       1,  "south", 25.0),
            _team("Houston",       2,  "south", 20.5),
            _team("Illinois",      3,  "south", 17.5),
            _team("Nebraska",      4,  "south", 15.0),
            _team("Vanderbilt",    5,  "south", 13.0),
            _team("N Carolina",    6,  "south", 11.0),
            _team("St Marys",      7,  "south",  9.0),
            _team("Clemson",       8,  "south",  7.0),
            _team("Iowa",          9,  "south",  5.0),
            _team("Texas AM",      10, "south",  3.0),
            _team("VCU",           11, "south",  1.0),
            _team("McNeese",       12, "south", -1.0),
            _team("Troy",          13, "south", -3.0),
            _team("Penn",          14, "south", -5.0),
            _team("Idaho",         15, "south", -7.0),
            _team("PV Lehigh",     16, "south",-10.0),
        ],
        "west": [
            _team("Arizona",       1,  "west",  26.0),
            _team("Purdue",        2,  "west",  21.0),
            _team("Gonzaga",       3,  "west",  18.0),
            _team("Arkansas",      4,  "west",  15.5),
            _team("Wisconsin",     5,  "west",  12.5),
            _team("BYU",           6,  "west",  10.5),
            _team("Miami",         7,  "west",   8.5),
            _team("Villanova",     8,  "west",   6.5),
            _team("Utah State",    9,  "west",   4.5),
            _team("Missouri",      10, "west",   2.5),
            _team("Tx NC State",   11, "west",   0.5),
            _team("High Point",    12, "west",  -1.5),
            _team("Hawaii",        13, "west",  -3.5),
            _team("Kennesaw St",   14, "west",  -5.5),
            _team("Queens",        15, "west",  -7.5),
            _team("LIU",           16, "west",  -9.5),
        ],
        "midwest": [
            _team("Michigan",      1,  "midwest", 27.5),
            _team("Iowa State",    2,  "midwest", 22.0),
            _team("Virginia",      3,  "midwest", 18.5),
            _team("Alabama",       4,  "midwest", 16.0),
            _team("Texas Tech",    5,  "midwest", 12.0),
            _team("Tennessee",     6,  "midwest", 10.0),
            _team("Kentucky",      7,  "midwest",  8.0),
            _team("Georgia",       8,  "midwest",  6.0),
            _team("St Louis",      9,  "midwest",  4.0),
            _team("Santa Clara",   10, "midwest",  2.0),
            _team("Miami OH SMU",  11, "midwest",  0.5),
            _team("Akron",         12, "midwest", -2.0),
            _team("Hofstra",       13, "midwest", -4.0),
            _team("Wright State",  14, "midwest", -6.0),
            _team("Tenn State",    15, "midwest", -8.0),
            _team("UMBC Howard",   16, "midwest", -9.5),
        ],
    }
    return regions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoolOptimalBracketStructure:

    def test_returns_dict_with_required_keys(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        assert "regions" in result
        assert "upsets" in result
        assert "pool_rationale" in result

    def test_all_four_regions_present(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        assert set(result["regions"].keys()) == {"east", "south", "west", "midwest"}

    def test_each_region_has_rounds_and_winner(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            assert "rounds" in data, f"{region} missing rounds"
            assert "winner" in data, f"{region} missing winner"
            assert data["winner"] is not None, f"{region} winner is None"

    def test_region_rounds_contain_correct_round_numbers(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            for rnd in [0, 1, 2, 3, 4]:
                assert rnd in data["rounds"], f"{region} missing round {rnd}"

    def test_r64_round_has_eight_matchups(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            assert len(data["rounds"][1]) == 8, f"{region} R64 should have 8 matchups"

    def test_elite_eight_has_one_matchup(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            assert len(data["rounds"][4]) == 1, f"{region} E8 should have 1 matchup"


class TestPoolOptimalBracketUpsetLogic:

    def test_default_produces_exactly_three_upsets(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        # 2 × 12v5 + 1 × 11v6 = 3 forced upsets
        assert len(result["upsets"]) == 3

    def test_upsets_are_r64_only(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for upset in result["upsets"]:
            assert upset["round"] == 1, f"Expected R64 upset, got round {upset['round']}"

    def test_all_upsets_are_double_digit_seeds(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for upset in result["upsets"]:
            assert upset["winner_seed"] >= 10, (
                f"Expected double-digit seed upset winner, got #{upset['winner_seed']}"
            )

    def test_no_duplicate_upsets(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        keys = [(u["region"], u["winner_seed"]) for u in result["upsets"]]
        assert len(keys) == len(set(keys)), "Duplicate upset entries found"

    def test_all_one_seeds_win_regions(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            winner = data["winner"]
            assert winner.seed == 1, (
                f"{region}: expected #1 seed to win region, got #{winner.seed} {winner.name}"
            )

    def test_zero_12v5_picks_produces_only_11v6_upsets(self):
        result = generate_pool_optimal_bracket(_make_bracket(), n_12v5_picks=0, n_11v6_picks=1)
        seeds = [u["winner_seed"] for u in result["upsets"]]
        assert all(s == 11 for s in seeds), "Expected only 11-seed upsets when 12v5=0"

    def test_zero_upsets_produces_chalk_r64(self):
        result = generate_pool_optimal_bracket(_make_bracket(), n_12v5_picks=0, n_11v6_picks=0)
        assert len(result["upsets"]) == 0, "Expected no upsets in full-chalk mode"

    def test_custom_n_picks_respected(self):
        result = generate_pool_optimal_bracket(_make_bracket(), n_12v5_picks=1, n_11v6_picks=0)
        assert len(result["upsets"]) == 1
        assert result["upsets"][0]["winner_seed"] == 12

    def test_each_upset_has_probability_between_zero_and_one(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for u in result["upsets"]:
            assert 0 < u["upset_prob"] < 1, f"Probability out of range: {u['upset_prob']}"

    def test_upsets_spread_across_different_regions(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        regions_used = {u["region"] for u in result["upsets"]}
        assert len(regions_used) >= 2, "Expected upsets in at least 2 different regions"


class TestPoolOptimalBracketRationale:

    def test_rationale_is_a_list(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        assert isinstance(result["pool_rationale"], list)

    def test_rationale_count_matches_upset_count(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        assert len(result["pool_rationale"]) == len(result["upsets"])

    def test_rationale_strings_contain_team_names(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for i, r in enumerate(result["pool_rationale"]):
            assert len(r) > 20, f"Rationale {i} is too short: {r!r}"

    def test_rationale_contains_historical_rate(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for r in result["pool_rationale"]:
            assert "%" in r, f"Rationale missing percentage: {r!r}"


class TestPoolOptimalBracketMatchupData:

    def test_each_matchup_has_winner_and_loser(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            for rnd in [1, 2, 3, 4]:
                for m in data["rounds"][rnd]:
                    assert m["winner"] is not None
                    assert m["loser"] is not None
                    assert m["winner"] is not m["loser"]

    def test_winner_is_ta_or_tb(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            for rnd in [1, 2, 3, 4]:
                for m in data["rounds"][rnd]:
                    assert m["winner"] is m["ta"] or m["winner"] is m["tb"]

    def test_is_upset_flag_correct(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            for rnd in [1, 2, 3, 4]:
                for m in data["rounds"][rnd]:
                    expected = m["winner"].seed > m["loser"].seed
                    assert m["is_upset"] == expected

    def test_win_prob_in_valid_range(self):
        result = generate_pool_optimal_bracket(_make_bracket())
        for region, data in result["regions"].items():
            for rnd in [1, 2, 3, 4]:
                for m in data["rounds"][rnd]:
                    assert 0 < m["prob"] < 1, f"prob={m['prob']} out of range"
