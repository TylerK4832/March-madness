"""Score model brackets against actual tournament results.

Uses the real bracket structure from MNCAATourneySlots.csv to:
1. Fill out a bracket using model probabilities
2. Replay the actual tournament
3. Score the bracket using ESPN-style point values
"""

import pandas as pd
import numpy as np
from config import BRACKET_SCORING, MC_SIMULATIONS, DATA_DIR


def load_bracket_structure(season: int) -> pd.DataFrame:
    """Load the bracket slot structure for a given season."""
    slots = pd.read_csv(DATA_DIR / "MNCAATourneySlots.csv")
    return slots[slots["Season"] == season].copy()


def load_actual_results(season: int) -> dict[tuple[int, int], int]:
    """Load actual tournament results. Returns {(teamA, teamB): winner_id}."""
    results = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    season_results = results[results["Season"] == season]
    game_results = {}
    for _, row in season_results.iterrows():
        key = (min(row["WTeamID"], row["LTeamID"]),
               max(row["WTeamID"], row["LTeamID"]))
        game_results[key] = row["WTeamID"]
    return game_results


def build_seed_to_team(season: int) -> dict[str, int]:
    """Map seed strings (e.g., 'W01') to team IDs."""
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    season_seeds = seeds[seeds["Season"] == season]
    return dict(zip(season_seeds["Seed"], season_seeds["TeamID"]))


def _slot_round(slot: str) -> int:
    """Extract round number from slot name. Play-in slots return 0."""
    if slot.startswith("R"):
        return int(slot[1])
    return 0  # play-in games (W16, X11, Y11, Y16, etc.)


def resolve_bracket(slots_df: pd.DataFrame, seed_to_team: dict[str, int],
                    matchup_probs: dict[tuple[int, int], float],
                    strategy: str = "chalk",
                    n_simulations: int = MC_SIMULATIONS) -> dict[str, int]:
    """Fill out a bracket by resolving each slot to a predicted winner.

    Args:
        slots_df: Bracket structure for the season.
        seed_to_team: Mapping of seed strings to team IDs.
        matchup_probs: {(teamA, teamB): P(teamA wins)} where teamA < teamB.
        strategy: 'chalk' (always pick higher prob) or 'pool' (MC optimized).
        n_simulations: Number of MC simulations for pool strategy.

    Returns:
        Dict mapping slot name to predicted winner team ID.
    """
    picks = {}  # slot -> team_id

    def resolve_team(ref: str) -> int | None:
        """Resolve a seed or slot reference to a team ID."""
        if ref in seed_to_team:
            return seed_to_team[ref]
        return picks.get(ref)

    def get_prob(team_a_id: int, team_b_id: int) -> float:
        low, high = min(team_a_id, team_b_id), max(team_a_id, team_b_id)
        return matchup_probs.get((low, high), 0.5)

    # Sort slots by round so we resolve play-ins first, then R1, R2, etc.
    slots_sorted = slots_df.copy()
    slots_sorted["round"] = slots_sorted["Slot"].apply(_slot_round)
    slots_sorted = slots_sorted.sort_values("round")

    if strategy == "chalk":
        for _, row in slots_sorted.iterrows():
            team_a = resolve_team(row["StrongSeed"])
            team_b = resolve_team(row["WeakSeed"])
            if team_a is None or team_b is None:
                continue
            prob_a = get_prob(team_a, team_b)
            # For chalk, "prob_a" is P(lower_id wins). Determine who is lower_id.
            low, high = min(team_a, team_b), max(team_a, team_b)
            if prob_a >= 0.5:
                picks[row["Slot"]] = low
            else:
                picks[row["Slot"]] = high

    elif strategy == "pool":
        # First, simulate many tournaments to estimate how often each team
        # reaches each slot
        rng = np.random.RandomState(42)
        slot_counts = {row["Slot"]: {} for _, row in slots_sorted.iterrows()}

        for _ in range(n_simulations):
            sim_picks = {}
            for _, row in slots_sorted.iterrows():
                ref_a = row["StrongSeed"]
                ref_b = row["WeakSeed"]
                ta = seed_to_team.get(ref_a) if ref_a in seed_to_team else sim_picks.get(ref_a)
                tb = seed_to_team.get(ref_b) if ref_b in seed_to_team else sim_picks.get(ref_b)
                if ta is None or tb is None:
                    continue
                low, high = min(ta, tb), max(ta, tb)
                prob_low = matchup_probs.get((low, high), 0.5)
                winner = low if rng.random() < prob_low else high
                sim_picks[row["Slot"]] = winner
                slot_counts[row["Slot"]][winner] = slot_counts[row["Slot"]].get(winner, 0) + 1

        # Now pick the team with highest expected value per slot
        for _, row in slots_sorted.iterrows():
            team_a = resolve_team(row["StrongSeed"])
            team_b = resolve_team(row["WeakSeed"])
            if team_a is None or team_b is None:
                continue
            rd = _slot_round(row["Slot"])
            round_points = BRACKET_SCORING.get(rd, 10)
            counts = slot_counts.get(row["Slot"], {})
            ev_a = counts.get(team_a, 0) / n_simulations * round_points
            ev_b = counts.get(team_b, 0) / n_simulations * round_points
            winner = team_a if ev_a >= ev_b else team_b
            picks[row["Slot"]] = winner

    return picks


def resolve_actual(slots_df: pd.DataFrame, seed_to_team: dict[str, int],
                   actual_results: dict[tuple[int, int], int]) -> dict[str, int]:
    """Resolve actual tournament winners for each slot."""
    actuals = {}

    def resolve_team(ref: str) -> int | None:
        if ref in seed_to_team:
            return seed_to_team[ref]
        return actuals.get(ref)

    slots_sorted = slots_df.copy()
    slots_sorted["round"] = slots_sorted["Slot"].apply(_slot_round)
    slots_sorted = slots_sorted.sort_values("round")

    for _, row in slots_sorted.iterrows():
        team_a = resolve_team(row["StrongSeed"])
        team_b = resolve_team(row["WeakSeed"])
        if team_a is None or team_b is None:
            continue
        key = (min(team_a, team_b), max(team_a, team_b))
        winner = actual_results.get(key)
        if winner is not None:
            actuals[row["Slot"]] = winner

    return actuals


def score_bracket(picks: dict[str, int], actuals: dict[str, int],
                  scoring: dict[int, int] = BRACKET_SCORING) -> dict:
    """Score a bracket against actual results.

    Returns:
        Dict with total_points, per_round breakdown, and max_possible.
    """
    round_correct = {}
    round_total = {}
    round_points = {}

    for slot, actual_winner in actuals.items():
        rd = _slot_round(slot)
        if rd == 0:
            continue  # skip play-in games (not scored in standard brackets)

        round_total[rd] = round_total.get(rd, 0) + 1
        pts = scoring.get(rd, 10)

        if picks.get(slot) == actual_winner:
            round_correct[rd] = round_correct.get(rd, 0) + 1
            round_points[rd] = round_points.get(rd, 0) + pts

    total_points = sum(round_points.values())
    max_points = sum(scoring.get(rd, 10) * count for rd, count in round_total.items())

    per_round = []
    for rd in sorted(round_total.keys()):
        per_round.append({
            "round": rd,
            "correct": round_correct.get(rd, 0),
            "total": round_total[rd],
            "points": round_points.get(rd, 0),
            "max_points": scoring.get(rd, 10) * round_total[rd],
        })

    return {
        "total_points": total_points,
        "max_points": max_points,
        "pct_of_max": total_points / max_points if max_points > 0 else 0,
        "per_round": per_round,
    }


def backtest_bracket(
    model,
    team_stats: pd.DataFrame,
    feature_cols: list[str],
    season: int,
    strategy: str = "chalk",
) -> dict:
    """End-to-end bracket backtest for one season.

    Args:
        model: Trained model with predict_proba().
        team_stats: DataFrame with team-season stats (including seeds, Massey).
        feature_cols: Feature columns to use.
        season: Season to backtest.
        strategy: 'chalk' or 'pool'.

    Returns:
        Scoring results dict.
    """
    from src.data_loader import load_seeds
    from predict import build_tournament_matchup_probs

    # Get tournament teams
    seeds_df = load_seeds()
    season_seeds = seeds_df[seeds_df["Season"] == season]
    tourney_teams = sorted(season_seeds["TeamID"].unique())
    seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

    # Build matchup probabilities
    matchup_probs = build_tournament_matchup_probs(
        model, team_stats, feature_cols, season, tourney_teams
    )

    # Load bracket structure
    slots_df = load_bracket_structure(season)
    seed_to_team = build_seed_to_team(season)

    # Fill bracket
    picks = resolve_bracket(slots_df, seed_to_team, matchup_probs, strategy=strategy)

    # Get actual results
    actual_results = load_actual_results(season)
    actuals = resolve_actual(slots_df, seed_to_team, actual_results)

    # Score
    return score_bracket(picks, actuals)
