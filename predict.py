"""Generate predictions and brackets for a specific season.

Usage:
    python predict.py --season 2024 --model xgboost --tier full
    python predict.py --season 2024 --model xgboost --tier full --bracket pool
"""

import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from config import FEATURE_TIERS
from src.feature_engineering import build_team_season_stats, add_seed_features, add_massey_features
from src.data_loader import load_seeds
from src.models import MODEL_REGISTRY
from src.bracket import generate_chalk_bracket, generate_pool_optimized_bracket


def build_tournament_matchup_probs(model, team_stats, feature_cols, season, tourney_team_ids):
    """Build probability predictions for all pairwise matchups among tournament teams."""
    season_teams = team_stats[team_stats["Season"] == season]
    seeds_df = load_seeds()
    season_seeds = seeds_df[seeds_df["Season"] == season]
    seed_lookup = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

    # Pre-index team stats
    stats_lookup = {}
    for tid in tourney_team_ids:
        rows = season_teams[season_teams["TeamID"] == tid]
        if len(rows) > 0:
            stats_lookup[tid] = rows.iloc[0]

    # Build all features in a batch
    batch_rows = []
    pair_keys = []
    for i, team_a in enumerate(tourney_team_ids):
        if team_a not in stats_lookup:
            continue
        for team_b in tourney_team_ids[i + 1:]:
            if team_b not in stats_lookup:
                continue

            a_stats = stats_lookup[team_a]
            b_stats = stats_lookup[team_b]
            a_seed = seed_lookup.get(team_a, 8)
            b_seed = seed_lookup.get(team_b, 8)

            row = {"SeedDiff": b_seed - a_seed}

            for stat in ["WinPct", "ScoreMargin", "FGPct", "FG3Pct", "FTPct",
                          "OR", "DR", "Ast", "TO", "Stl", "Blk"]:
                a_val = a_stats.get(stat, 0)
                b_val = b_stats.get(stat, 0)
                row[f"{stat}Diff"] = (a_val if not pd.isna(a_val) else 0) - (b_val if not pd.isna(b_val) else 0)

            for col in ["OrdinalKenPom", "OrdinalSagarin"]:
                a_val = a_stats.get(col, 180)
                b_val = b_stats.get(col, 180)
                row[f"{col}Diff"] = (b_val if not pd.isna(b_val) else 180) - (a_val if not pd.isna(a_val) else 180)

            for stat in ["TOMargin", "RebMargin", "SoSProxy",
                         "OE", "DE", "NetEff", "eFGPct", "TOPct", "ORPct", "FTRate",
                         "AdjOE", "AdjDE", "AdjNetEff"]:
                a_val = a_stats.get(stat, 0)
                b_val = b_stats.get(stat, 0)
                row[f"{stat}Diff"] = (a_val if not pd.isna(a_val) else 0) - (b_val if not pd.isna(b_val) else 0)

            batch_rows.append(row)
            pair_keys.append((team_a, team_b))

    # Predict all at once
    available = [c for c in feature_cols if c in batch_rows[0]]
    X = pd.DataFrame(batch_rows)[available]
    probs = model.predict_proba(X)

    return dict(zip(pair_keys, probs))


def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Prediction")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--tier", type=str, default="full")
    parser.add_argument("--bracket", type=str, default="chalk",
                        choices=["chalk", "pool"])
    args = parser.parse_args()

    from src.feature_engineering import build_matchup_features

    feature_cols = FEATURE_TIERS[args.tier]

    # Train on all seasons except the target
    print(f"Building features (tier: {args.tier})...")
    X, y, seasons = build_matchup_features(args.tier)

    train_mask = seasons != args.season
    X_train = X[train_mask]
    y_train = y[train_mask]

    print(f"Training {args.model} on {len(X_train)} games...")
    model_cls = MODEL_REGISTRY[args.model]
    model_kwargs = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05} if args.model == "xgboost" else {}
    model = model_cls(**model_kwargs)
    model.fit(X_train, y_train)

    # Build team stats for the target season
    team_stats = build_team_season_stats()
    team_stats = add_seed_features(team_stats)
    team_stats = add_massey_features(team_stats)

    # Get tournament teams for target season
    seeds_df = load_seeds()
    season_seeds = seeds_df[seeds_df["Season"] == args.season]
    if len(season_seeds) == 0:
        print(f"No seed data found for season {args.season}")
        return

    seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
    tourney_teams = sorted(season_seeds["TeamID"].unique())

    # Build pairwise probabilities (only among tournament teams)
    print(f"Computing matchup probabilities for {len(tourney_teams)} teams...")
    matchup_probs = build_tournament_matchup_probs(
        model, team_stats, feature_cols, args.season, tourney_teams
    )
    print(f"  {len(matchup_probs)} matchup probabilities computed.")

    # Order teams by seed for bracket
    bracket_teams = sorted(tourney_teams, key=lambda t: seed_map.get(t, 16))[:64]

    # Generate bracket
    if args.bracket == "chalk":
        print("\nCHALK BRACKET (always pick higher probability):")
        results = generate_chalk_bracket(matchup_probs, seed_map, bracket_teams)
    else:
        print("\nPOOL-OPTIMIZED BRACKET (maximize expected score):")
        results = generate_pool_optimized_bracket(matchup_probs, seed_map, bracket_teams)

    # Display
    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
                   4: "Elite 8", 5: "Final Four", 6: "Championship"}
    current_round = 0
    for r in results:
        if r["round"] != current_round:
            current_round = r["round"]
            print(f"\n  {round_names.get(current_round, f'Round {current_round}')}:")
        prob_display = r["prob_a_wins"]
        print(f"    ({seed_map.get(r['team_a'], '?'):>2}) Team {r['team_a']} vs "
              f"({seed_map.get(r['team_b'], '?'):>2}) Team {r['team_b']}  "
              f"-> Pick: Team {r['pick']} (seed {r['pick_seed']}, "
              f"p={prob_display:.3f})")


if __name__ == "__main__":
    main()
