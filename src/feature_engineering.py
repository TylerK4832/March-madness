"""Build team-season aggregates and matchup differentials."""

import pandas as pd
import numpy as np
from src.data_loader import (
    load_regular_season_detailed,
    load_seeds,
    load_tourney_detailed,
    load_massey_ordinals,
)


def build_team_season_stats() -> pd.DataFrame:
    """Aggregate regular season box scores into per-game averages per team-season."""
    rs = load_regular_season_detailed()

    # Build stats from the winner perspective
    w_stats = rs.rename(columns=lambda c: c[1:] if c.startswith("W") and c != "WLoc" else c)
    w_stats = w_stats.rename(columns={"TeamID": "WTeamID"})
    w_stats["Won"] = 1
    # Rename losing team columns to opponent prefix
    l_cols = {c: "Opp" + c[1:] for c in rs.columns if c.startswith("L") and c != "LTeamID"}
    w_stats = w_stats.rename(columns=l_cols)

    # Build stats from the loser perspective
    l_stats = rs.copy()
    l_rename = {c: c[1:] for c in rs.columns if c.startswith("L") and c != "LTeamID"}
    l_rename["LTeamID"] = "WTeamID"  # We'll rename to TeamID later
    w_rename = {c: "Opp" + c[1:] for c in rs.columns if c.startswith("W") and c not in ("WLoc", "WTeamID")}
    w_rename["WTeamID"] = "OppTeamID"
    l_stats = l_stats.rename(columns={**l_rename, **w_rename})
    l_stats["Won"] = 0

    # Actually let's take a cleaner approach
    # For each game, compute stats from both teams' perspectives
    records = []
    for _, row in rs.iterrows():
        season = row["Season"]
        # Winner record
        records.append({
            "Season": season, "TeamID": row["WTeamID"],
            "Won": 1,
            "Score": row["WScore"], "OppScore": row["LScore"],
            "FGM": row["WFGM"], "FGA": row["WFGA"],
            "FGM3": row["WFGM3"], "FGA3": row["WFGA3"],
            "FTM": row["WFTM"], "FTA": row["WFTA"],
            "OR": row["WOR"], "DR": row["WDR"],
            "Ast": row["WAst"], "TO": row["WTO"],
            "Stl": row["WStl"], "Blk": row["WBlk"],
            "OppOR": row["LOR"], "OppDR": row["LDR"],
            "OppTO": row["LTO"],
        })
        # Loser record
        records.append({
            "Season": season, "TeamID": row["LTeamID"],
            "Won": 0,
            "Score": row["LScore"], "OppScore": row["WScore"],
            "FGM": row["LFGM"], "FGA": row["LFGA"],
            "FGM3": row["LFGM3"], "FGA3": row["LFGA3"],
            "FTM": row["LFTM"], "FTA": row["LFTA"],
            "OR": row["LOR"], "DR": row["LDR"],
            "Ast": row["LAst"], "TO": row["LTO"],
            "Stl": row["LStl"], "Blk": row["LBlk"],
            "OppOR": row["WOR"], "OppDR": row["WDR"],
            "OppTO": row["WTO"],
        })

    game_df = pd.DataFrame(records)

    # Aggregate to team-season level
    agg = game_df.groupby(["Season", "TeamID"]).agg(
        Games=("Won", "count"),
        Wins=("Won", "sum"),
        Score=("Score", "mean"),
        OppScore=("OppScore", "mean"),
        FGM=("FGM", "mean"),
        FGA=("FGA", "mean"),
        FGM3=("FGM3", "mean"),
        FGA3=("FGA3", "mean"),
        FTM=("FTM", "mean"),
        FTA=("FTA", "mean"),
        OR=("OR", "mean"),
        DR=("DR", "mean"),
        Ast=("Ast", "mean"),
        TO=("TO", "mean"),
        Stl=("Stl", "mean"),
        Blk=("Blk", "mean"),
        OppOR=("OppOR", "mean"),
        OppDR=("OppDR", "mean"),
        OppTO=("OppTO", "mean"),
    ).reset_index()

    # Derived features
    agg["WinPct"] = agg["Wins"] / agg["Games"]
    agg["ScoreMargin"] = agg["Score"] - agg["OppScore"]
    agg["FGPct"] = agg["FGM"] / agg["FGA"].clip(lower=1)
    agg["FG3Pct"] = agg["FGM3"] / agg["FGA3"].clip(lower=1)
    agg["FTPct"] = agg["FTM"] / agg["FTA"].clip(lower=1)
    agg["TOMargin"] = agg["OppTO"] - agg["TO"]  # positive = good
    agg["RebMargin"] = (agg["OR"] + agg["DR"]) - (agg["OppOR"] + agg["OppDR"])

    # Strength of schedule proxy: average opponent scoring margin
    # (will be zero-ish by construction for synthetic data but works with real data)
    sos = game_df.groupby(["Season", "TeamID"]).agg(
        SoSProxy=("OppScore", "mean")
    ).reset_index()
    agg = agg.merge(sos, on=["Season", "TeamID"], how="left")

    return agg


def add_seed_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge tournament seed numbers into team stats."""
    seeds = load_seeds()[["Season", "TeamID", "SeedNum"]]
    return team_stats.merge(seeds, on=["Season", "TeamID"], how="left")


def add_massey_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge Massey ordinal rankings (KenPom, Sagarin) into team stats."""
    massey = load_massey_ordinals()

    # Pivot: one column per ranking system
    pivot = massey.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"POM": "OrdinalKenPom", "SAG": "OrdinalSagarin"})

    return team_stats.merge(pivot, on=["Season", "TeamID"], how="left")


def build_matchup_features(feature_tier: str = "full") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the full training dataset: matchup differentials for tournament games.

    Returns:
        X: DataFrame of features (differentials)
        y: Series of labels (1 if lower TeamID won)
        seasons: Series of season for each row (for CV splits)
    """
    from config import FEATURE_TIERS

    # Build team-season aggregates
    team_stats = build_team_season_stats()
    team_stats = add_seed_features(team_stats)
    team_stats = add_massey_features(team_stats)

    # Load tournament games
    tourney = load_tourney_detailed()

    rows = []
    for _, game in tourney.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        # Symmetric encoding: lower ID is always TeamA
        team_a = min(w_id, l_id)
        team_b = max(w_id, l_id)
        label = 1 if w_id == team_a else 0

        a_stats = team_stats[(team_stats["Season"] == season) &
                             (team_stats["TeamID"] == team_a)]
        b_stats = team_stats[(team_stats["Season"] == season) &
                             (team_stats["TeamID"] == team_b)]

        if a_stats.empty or b_stats.empty:
            continue

        a = a_stats.iloc[0]
        b = b_stats.iloc[0]

        row = {"Season": season, "TeamA": team_a, "TeamB": team_b, "Label": label}

        # Seed differential
        a_seed = a.get("SeedNum", 8)
        b_seed = b.get("SeedNum", 8)
        if pd.isna(a_seed):
            a_seed = 8
        if pd.isna(b_seed):
            b_seed = 8
        row["SeedDiff"] = b_seed - a_seed  # positive = A has better seed

        # Box score differentials
        for stat in ["WinPct", "ScoreMargin", "FGPct", "FG3Pct", "FTPct",
                      "OR", "DR", "Ast", "TO", "Stl", "Blk"]:
            a_val = a.get(stat, 0)
            b_val = b.get(stat, 0)
            if pd.isna(a_val):
                a_val = 0
            if pd.isna(b_val):
                b_val = 0
            row[f"{stat}Diff"] = a_val - b_val

        # Massey ordinal differentials (lower rank = better, so B - A)
        for col in ["OrdinalKenPom", "OrdinalSagarin"]:
            a_val = a.get(col, 180)
            b_val = b.get(col, 180)
            if pd.isna(a_val):
                a_val = 180
            if pd.isna(b_val):
                b_val = 180
            row[f"{col}Diff"] = b_val - a_val  # positive = A ranked better

        # Derived differentials
        for stat in ["TOMargin", "RebMargin", "SoSProxy"]:
            a_val = a.get(stat, 0)
            b_val = b.get(stat, 0)
            if pd.isna(a_val):
                a_val = 0
            if pd.isna(b_val):
                b_val = 0
            row[f"{stat}Diff"] = a_val - b_val

        rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = FEATURE_TIERS[feature_tier]
    # Only use columns that exist
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].astype(float)
    y = df["Label"].astype(int)
    seasons = df["Season"]

    return X, y, seasons
