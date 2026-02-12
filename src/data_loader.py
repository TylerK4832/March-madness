"""Load and parse Kaggle March Madness CSV files."""

import pandas as pd
from config import DATA_DIR, FIRST_DETAILED_SEASON, TOURNEY_START_DAY


def load_teams() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "MTeams.csv")


def load_seeds() -> pd.DataFrame:
    """Load tournament seeds and parse numeric seed value."""
    df = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    df["SeedNum"] = df["Seed"].apply(
        lambda s: int("".join(c for c in s[1:] if c.isdigit()))
    )
    return df


def load_regular_season_detailed() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    return df[df["Season"] >= FIRST_DETAILED_SEASON]


def load_tourney_detailed() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    return df[df["Season"] >= FIRST_DETAILED_SEASON]


def load_massey_ordinals(systems=("POM", "SAG")) -> pd.DataFrame:
    """Load Massey ordinals, keeping only specified systems near tournament start."""
    df = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
    df = df[df["SystemName"].isin(systems)]
    df = df[df["Season"] >= FIRST_DETAILED_SEASON]

    # Keep the latest ranking day before tournament for each season/system/team
    df = df[df["RankingDayNum"] <= TOURNEY_START_DAY]
    idx = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
    return df.loc[idx].reset_index(drop=True)
