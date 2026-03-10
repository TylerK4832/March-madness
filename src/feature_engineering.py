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

    # For each game, compute stats from both teams' perspectives
    records = []
    for _, row in rs.iterrows():
        season = row["Season"]
        for is_winner in [True, False]:
            p = "W" if is_winner else "L"
            o = "L" if is_winner else "W"

            fga = row[f"{p}FGA"]
            fta = row[f"{p}FTA"]
            orb = row[f"{p}OR"]
            to = row[f"{p}TO"]
            ofga = row[f"{o}FGA"]
            ofta = row[f"{o}FTA"]
            oorb = row[f"{o}OR"]
            oto = row[f"{o}TO"]

            # Estimate possessions (Dean Oliver formula)
            orb_pct = orb / max(orb + row[f"{o}DR"], 1)
            oorb_pct = oorb / max(oorb + row[f"{p}DR"], 1)
            poss = 0.5 * (
                (fga + 0.4 * fta - 1.07 * orb_pct * (fga - row[f"{p}FGM"]) + to)
                + (ofga + 0.4 * ofta - 1.07 * oorb_pct * (ofga - row[f"{o}FGM"]) + oto)
            )
            if poss < 1:
                poss = 60

            records.append({
                "Season": season, "DayNum": row["DayNum"],
                "TeamID": row[f"{p}TeamID"],
                "OppID": row[f"{o}TeamID"],
                "Won": 1 if is_winner else 0,
                "Score": row[f"{p}Score"], "OppScore": row[f"{o}Score"],
                "FGM": row[f"{p}FGM"], "FGA": fga,
                "FGM3": row[f"{p}FGM3"], "FGA3": row[f"{p}FGA3"],
                "FTM": row[f"{p}FTM"], "FTA": fta,
                "OR": orb, "DR": row[f"{p}DR"],
                "Ast": row[f"{p}Ast"], "TO": to,
                "Stl": row[f"{p}Stl"], "Blk": row[f"{p}Blk"],
                "OppOR": oorb, "OppDR": row[f"{o}DR"],
                "OppTO": oto,
                # Tempo-adjusted stats
                "Poss": poss,
                "OE": row[f"{p}Score"] / poss * 100,
                "DE": row[f"{o}Score"] / poss * 100,
                "eFGPct": (row[f"{p}FGM"] + 0.5 * row[f"{p}FGM3"]) / max(fga, 1),
                "TOPct": to / max(poss, 1),
                "ORPct": orb / max(orb + row[f"{o}DR"], 1),
                "FTRate": fta / max(fga, 1),
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
        Poss=("Poss", "mean"),
        OE=("OE", "mean"),
        DE=("DE", "mean"),
        eFGPct=("eFGPct", "mean"),
        TOPct=("TOPct", "mean"),
        ORPct=("ORPct", "mean"),
        FTRate=("FTRate", "mean"),
    ).reset_index()

    # Derived features
    agg["WinPct"] = agg["Wins"] / agg["Games"]
    agg["ScoreMargin"] = agg["Score"] - agg["OppScore"]
    agg["FGPct"] = agg["FGM"] / agg["FGA"].clip(lower=1)
    agg["FG3Pct"] = agg["FGM3"] / agg["FGA3"].clip(lower=1)
    agg["FTPct"] = agg["FTM"] / agg["FTA"].clip(lower=1)
    agg["TOMargin"] = agg["OppTO"] - agg["TO"]  # positive = good
    agg["RebMargin"] = (agg["OR"] + agg["DR"]) - (agg["OppOR"] + agg["OppDR"])
    agg["NetEff"] = agg["OE"] - agg["DE"]

    # Strength of schedule proxy: average opponent scoring margin
    # (will be zero-ish by construction for synthetic data but works with real data)
    sos = game_df.groupby(["Season", "TeamID"]).agg(
        SoSProxy=("OppScore", "mean")
    ).reset_index()
    agg = agg.merge(sos, on=["Season", "TeamID"], how="left")

    # Opponent-adjusted efficiency
    adj = compute_adjusted_efficiency(game_df)
    agg = agg.merge(adj, on=["Season", "TeamID"], how="left")

    # Late-season momentum: last 10 games before tournament
    late = compute_late_season_stats(game_df)
    agg = agg.merge(late, on=["Season", "TeamID"], how="left")

    # Consistency and close-game stats
    extra = compute_consistency_stats(game_df)
    agg = agg.merge(extra, on=["Season", "TeamID"], how="left")

    return agg


def compute_adjusted_efficiency(game_df: pd.DataFrame, n_iterations: int = 20) -> pd.DataFrame:
    """Compute opponent-adjusted offensive/defensive efficiency per team-season.

    Uses iterative adjustment: each team's raw efficiency is adjusted based on
    the quality of opponents faced. A team scoring 100 pts/100poss against a
    defense that allows 95 (below avg) is better than one scoring 105 against
    a defense that allows 115 (above avg).

    Similar in spirit to KenPom's adjusted efficiency.
    """
    results = []

    for season, sdf in game_df.groupby("Season"):
        teams = sdf["TeamID"].unique()

        league_oe = sdf["OE"].mean()
        league_de = sdf["DE"].mean()

        # Initialize ratings at league average
        adj_oe = {t: league_oe for t in teams}
        adj_de = {t: league_de for t in teams}

        for _ in range(n_iterations):
            new_oe = {t: [] for t in teams}
            new_de = {t: [] for t in teams}

            for _, g in sdf.iterrows():
                tid = g["TeamID"]
                oid = int(g.get("OppID", 0))
                if oid == 0:
                    # Need opponent ID - infer from game structure
                    continue
                if oid not in adj_de:
                    continue

                opp_de_factor = adj_de[oid] / league_de
                opp_oe_factor = adj_oe[oid] / league_oe

                # Adjust: what would this team score/allow vs average opponent?
                new_oe[tid].append(g["OE"] / opp_de_factor)
                new_de[tid].append(g["DE"] / opp_oe_factor)

            for t in teams:
                if new_oe[t]:
                    adj_oe[t] = np.mean(new_oe[t])
                if new_de[t]:
                    adj_de[t] = np.mean(new_de[t])

        for t in teams:
            results.append({
                "Season": season,
                "TeamID": t,
                "AdjOE": adj_oe[t],
                "AdjDE": adj_de[t],
                "AdjNetEff": adj_oe[t] - adj_de[t],
            })

    return pd.DataFrame(results)


def compute_late_season_stats(game_df: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Compute stats from the last N regular season games (momentum/form).

    Captures late-season form: teams peaking at the right time often outperform
    their season averages in March.
    """
    from config import TOURNEY_START_DAY

    results = []
    for (season, tid), gdf in game_df.groupby(["Season", "TeamID"]):
        # Only pre-tournament games
        pre_tourney = gdf[gdf["DayNum"] < TOURNEY_START_DAY].sort_values("DayNum")
        if len(pre_tourney) < 5:
            continue
        last_n = pre_tourney.tail(n_games)
        results.append({
            "Season": season,
            "TeamID": tid,
            "LateWinPct": last_n["Won"].mean(),
            "LateOE": last_n["OE"].mean(),
            "LateDE": last_n["DE"].mean(),
            "LateMargin": (last_n["Score"] - last_n["OppScore"]).mean(),
        })
    return pd.DataFrame(results)


def compute_consistency_stats(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute consistency and close-game performance metrics.

    - OE/DE variance: more consistent teams may be more reliable in single-elimination
    - Close game win%: teams that win close games may have a mental edge
    """
    results = []
    for (season, tid), gdf in game_df.groupby(["Season", "TeamID"]):
        margin = gdf["Score"] - gdf["OppScore"]
        close_mask = margin.abs() <= 5  # close games within 5 points
        close_games = gdf[close_mask]

        results.append({
            "Season": season,
            "TeamID": tid,
            "OEStd": gdf["OE"].std(),
            "MarginStd": margin.std(),
            "CloseWinPct": close_games["Won"].mean() if len(close_games) >= 3 else 0.5,
            "CloseGamePct": len(close_games) / len(gdf),  # fraction of games that are close
        })
    return pd.DataFrame(results)


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
        for stat in ["TOMargin", "RebMargin", "SoSProxy",
                      "OE", "DE", "NetEff", "eFGPct", "TOPct", "ORPct", "FTRate",
                      "AdjOE", "AdjDE", "AdjNetEff",
                      "LateWinPct", "LateOE", "LateDE", "LateMargin",
                      "OEStd", "MarginStd", "CloseWinPct", "CloseGamePct"]:
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
