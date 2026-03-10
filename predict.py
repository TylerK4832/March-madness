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
from src.feature_engineering import build_matchup_rows
from src.data_loader import load_seeds
from src.models import MODEL_REGISTRY
from src.pipeline import make_pipeline
from src.bracket import generate_chalk_bracket, generate_pool_optimized_bracket


def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Prediction")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--tier", type=str, default="full")
    parser.add_argument("--bracket", type=str, default="chalk",
                        choices=["chalk", "pool"])
    args = parser.parse_args()

    # Build data
    print(f"Building features (tier: {args.tier})...")
    matchup_df, team_stats = build_matchup_rows()

    # Train on all seasons except the target
    train_mask = matchup_df["Season"] != args.season
    X_train = matchup_df[train_mask][["Season", "TeamA", "TeamB"]]
    y_train = matchup_df[train_mask]["Label"].astype(int)

    print(f"Training {args.model} on {len(X_train)} games...")
    model_cls = MODEL_REGISTRY[args.model]
    model_kwargs = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05} if args.model == "xgboost" else {}
    pipe = make_pipeline(model_cls, model_kwargs, team_stats, args.tier)
    pipe.fit(X_train, y_train)

    # Get tournament teams for target season
    seeds_df = load_seeds()
    season_seeds = seeds_df[seeds_df["Season"] == args.season]
    if len(season_seeds) == 0:
        print(f"No seed data found for season {args.season}")
        return

    seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
    tourney_teams = sorted(season_seeds["TeamID"].unique())

    # Build all pairwise matchup rows
    print(f"Computing matchup probabilities for {len(tourney_teams)} teams...")
    pair_rows = []
    for i, team_a in enumerate(tourney_teams):
        for team_b in tourney_teams[i + 1:]:
            pair_rows.append({
                "Season": args.season,
                "TeamA": team_a,
                "TeamB": team_b,
            })

    X_pred = pd.DataFrame(pair_rows)
    probs = pipe.predict_proba(X_pred)[:, 1]

    matchup_probs = {}
    for idx, row in X_pred.iterrows():
        matchup_probs[(int(row["TeamA"]), int(row["TeamB"]))] = probs[idx]

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
