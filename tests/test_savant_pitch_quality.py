from backend.fantasy_baseball.savant_pitch_quality import (
    SavantPitcherInput,
    calculate_savant_pitch_quality,
    score_pitcher_population,
)


def test_high_skill_pitcher_gets_breakout_arm_signal():
    pitcher = SavantPitcherInput(
        player_id="111",
        player_name="Breakout Arm",
        season=2026,
        xera=2.85,
        xwoba=0.270,
        barrel_percent_allowed=5.0,
        hard_hit_percent_allowed=31.0,
        avg_exit_velocity_allowed=86.0,
        k_percent=31.0,
        bb_percent=6.5,
        k_9=11.4,
        whiff_percent=33.0,
        ip=42.0,
        pitches=690,
        era=3.65,
        whip=1.08,
    )

    score = calculate_savant_pitch_quality(pitcher)

    assert score.savant_pitch_quality >= 112
    assert score.sample_confidence >= 0.70
    assert "BREAKOUT_ARM" in score.signals
    assert "RATIO_RISK" not in score.signals


def test_low_sample_skill_change_is_watchlist_not_breakout_arm():
    pitcher = SavantPitcherInput(
        player_id="222",
        player_name="Tiny Sample Arm",
        season=2026,
        xera=3.05,
        xwoba=0.285,
        barrel_percent_allowed=6.0,
        hard_hit_percent_allowed=34.0,
        avg_exit_velocity_allowed=87.5,
        k_percent=30.0,
        bb_percent=8.0,
        k_9=10.8,
        whiff_percent=32.0,
        ip=7.0,
        pitches=115,
        era=4.50,
        whip=1.25,
    )

    score = calculate_savant_pitch_quality(pitcher)

    assert score.sample_confidence < 0.50
    assert "WATCHLIST" in score.signals
    assert "BREAKOUT_ARM" not in score.signals


def test_whiffs_with_bad_command_and_contact_are_ratio_risk():
    pitcher = SavantPitcherInput(
        player_id="333",
        player_name="Wild Stuff",
        season=2026,
        xera=4.60,
        xwoba=0.345,
        barrel_percent_allowed=12.5,
        hard_hit_percent_allowed=46.0,
        avg_exit_velocity_allowed=91.0,
        k_percent=29.0,
        bb_percent=13.0,
        k_9=10.5,
        whiff_percent=31.0,
        ip=35.0,
        pitches=620,
        era=3.40,
        whip=1.42,
    )

    score = calculate_savant_pitch_quality(pitcher)

    assert score.bat_missing_skill >= 108
    assert score.command_stability < 95
    assert score.contact_suppression < 95
    assert "RATIO_RISK" in score.signals
    assert "BREAKOUT_ARM" not in score.signals


def test_population_scoring_ranks_breakout_profile_above_ratio_risk():
    pitchers = [
        SavantPitcherInput(
            player_id="111",
            player_name="Breakout Arm",
            season=2026,
            xera=2.85,
            xwoba=0.270,
            barrel_percent_allowed=5.0,
            hard_hit_percent_allowed=31.0,
            avg_exit_velocity_allowed=86.0,
            k_percent=31.0,
            bb_percent=6.5,
            k_9=11.4,
            whiff_percent=33.0,
            ip=42.0,
            pitches=690,
            era=3.65,
            whip=1.08,
        ),
        SavantPitcherInput(
            player_id="333",
            player_name="Wild Stuff",
            season=2026,
            xera=4.60,
            xwoba=0.345,
            barrel_percent_allowed=12.5,
            hard_hit_percent_allowed=46.0,
            avg_exit_velocity_allowed=91.0,
            k_percent=29.0,
            bb_percent=13.0,
            k_9=10.5,
            whiff_percent=31.0,
            ip=35.0,
            pitches=620,
            era=3.40,
            whip=1.42,
        ),
        SavantPitcherInput(
            player_id="444",
            player_name="Average Arm",
            season=2026,
            xera=4.05,
            xwoba=0.315,
            barrel_percent_allowed=8.5,
            hard_hit_percent_allowed=39.0,
            avg_exit_velocity_allowed=88.8,
            k_percent=22.0,
            bb_percent=8.3,
            k_9=8.2,
            whiff_percent=23.0,
            ip=48.0,
            pitches=760,
            era=4.10,
            whip=1.28,
        ),
    ]

    scores = score_pitcher_population(pitchers)
    by_name = {score.player_name: score for score in scores}

    assert by_name["Breakout Arm"].savant_pitch_quality > by_name["Wild Stuff"].savant_pitch_quality
    assert by_name["Breakout Arm"].savant_pitch_quality > by_name["Average Arm"].savant_pitch_quality
    assert abs(by_name["Average Arm"].savant_pitch_quality - 100) <= 12
