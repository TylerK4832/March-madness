"""Bracket generation: chalk picks and Monte Carlo pool-optimized."""

import numpy as np
import pandas as pd
from config import BRACKET_SCORING, MC_SIMULATIONS


def generate_chalk_bracket(matchup_probs: dict[tuple[int, int], float],
                           seeds: dict[int, int],
                           teams: list[int]) -> list[dict]:
    """Generate bracket by always picking the higher-probability team.

    Args:
        matchup_probs: Dict mapping (teamA, teamB) -> P(teamA wins), where teamA < teamB.
        seeds: Dict mapping teamID -> seed number.
        teams: List of 64 team IDs in bracket order.

    Returns:
        List of dicts with round, matchup, and winner info.
    """
    bracket = list(teams)
    results = []
    round_num = 1

    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            if i + 1 >= len(bracket):
                next_round.append(bracket[i])
                continue

            t1, t2 = bracket[i], bracket[i + 1]
            team_a, team_b = min(t1, t2), max(t1, t2)
            prob_a = matchup_probs.get((team_a, team_b), 0.5)

            winner = team_a if prob_a >= 0.5 else team_b
            next_round.append(winner)
            results.append({
                "round": round_num,
                "team_a": team_a,
                "team_b": team_b,
                "prob_a_wins": prob_a,
                "pick": winner,
                "pick_seed": seeds.get(winner, "?"),
            })

        bracket = next_round
        round_num += 1

    return results


def simulate_tournament(matchup_probs: dict[tuple[int, int], float],
                        teams: list[int],
                        rng: np.random.RandomState) -> list[int]:
    """Simulate one tournament using the model's probabilities."""
    bracket = list(teams)
    round_winners = []

    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            if i + 1 >= len(bracket):
                next_round.append(bracket[i])
                continue

            t1, t2 = bracket[i], bracket[i + 1]
            team_a, team_b = min(t1, t2), max(t1, t2)
            prob_a = matchup_probs.get((team_a, team_b), 0.5)

            winner = team_a if rng.random() < prob_a else team_b
            next_round.append(winner)
            round_winners.append(winner)

        bracket = next_round

    return round_winners


def generate_pool_optimized_bracket(
    matchup_probs: dict[tuple[int, int], float],
    seeds: dict[int, int],
    teams: list[int],
    n_simulations: int = MC_SIMULATIONS,
    scoring: dict[int, int] = BRACKET_SCORING,
) -> list[dict]:
    """Generate bracket optimized for pool scoring via Monte Carlo.

    For each possible bracket, estimate expected score by simulating many
    tournaments and scoring each one. Pick the bracket that maximizes
    expected score under the weighted scoring rules.

    In practice, we pick each game independently by choosing the team
    that maximizes expected points for that slot.
    """
    rng = np.random.RandomState(42)

    # Simulate many tournaments
    all_sims = []
    for _ in range(n_simulations):
        winners = simulate_tournament(matchup_probs, teams, rng)
        all_sims.append(winners)

    # For each slot in the bracket, count how often each team reaches that slot
    n_games = len(teams) - 1
    slot_counts: dict[int, dict[int, int]] = {
        i: {} for i in range(n_games)
    }

    for sim in all_sims:
        for slot_idx, winner in enumerate(sim):
            slot_counts[slot_idx][winner] = slot_counts[slot_idx].get(winner, 0) + 1

    # Determine which round each slot belongs to
    slots_per_round = []
    remaining = len(teams)
    total_slots = 0
    round_num = 1
    while remaining > 1:
        games_in_round = remaining // 2
        for _ in range(games_in_round):
            slots_per_round.append(round_num)
            total_slots += 1
        remaining = remaining // 2
        round_num += 1

    # For each slot, pick the team most likely to reach it (weighted by round points)
    bracket = list(teams)
    results = []
    round_num = 1
    slot_idx = 0

    while len(bracket) > 1:
        next_round = []
        round_points = scoring.get(round_num, 10)

        for i in range(0, len(bracket), 2):
            if i + 1 >= len(bracket):
                next_round.append(bracket[i])
                continue

            t1, t2 = bracket[i], bracket[i + 1]
            team_a, team_b = min(t1, t2), max(t1, t2)
            prob_a = matchup_probs.get((team_a, team_b), 0.5)

            # Expected value: probability of reaching this point * round points
            # Use simulation frequency as proxy
            counts = slot_counts.get(slot_idx, {})
            ev_t1 = counts.get(t1, 0) / n_simulations * round_points
            ev_t2 = counts.get(t2, 0) / n_simulations * round_points

            winner = t1 if ev_t1 >= ev_t2 else t2
            next_round.append(winner)
            results.append({
                "round": round_num,
                "team_a": team_a,
                "team_b": team_b,
                "prob_a_wins": prob_a,
                "pick": winner,
                "pick_seed": seeds.get(winner, "?"),
                "ev_t1": ev_t1,
                "ev_t2": ev_t2,
            })
            slot_idx += 1

        bracket = next_round
        round_num += 1

    return results
