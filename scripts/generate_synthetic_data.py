"""Generate synthetic data matching the Kaggle March Machine Learning Mania schema.

This creates realistic-looking CSV files so the pipeline can be tested end-to-end
without needing the actual Kaggle dataset. Replace files in data/ with real Kaggle
data for actual predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"
SEASONS = list(range(2003, 2025))
NUM_TEAMS = 68  # Tournament-eligible teams (we'll create more for regular season)
TOTAL_TEAMS = 364


def generate_teams():
    """Generate MTeams.csv."""
    rows = []
    for tid in range(1100, 1100 + TOTAL_TEAMS):
        rows.append({"TeamID": tid, "TeamName": f"Team_{tid}"})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "MTeams.csv", index=False)
    return df


def generate_seeds():
    """Generate MNCAATourneySeeds.csv."""
    rng = np.random.RandomState(SEED)
    rows = []
    regions = ["W", "X", "Y", "Z"]
    for season in SEASONS:
        team_ids = list(range(1100, 1100 + TOTAL_TEAMS))
        selected = rng.choice(team_ids, size=NUM_TEAMS, replace=False)
        seed_num = 0
        for i, tid in enumerate(selected):
            region = regions[i % 4]
            seed_num = (i // 4) + 1
            if seed_num > 16:
                # Play-in seeds (16a/16b)
                seed_str = f"{region}{16:02d}a"
            else:
                seed_str = f"{region}{seed_num:02d}"
            rows.append({"Season": season, "Seed": seed_str, "TeamID": tid})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "MNCAATourneySeeds.csv", index=False)
    return df


def _generate_game_stats(rng, skill_w, skill_l):
    """Generate realistic box score stats for one game."""
    # Score influenced by skill
    w_score = int(np.clip(rng.normal(75 + skill_w * 5, 10), 50, 120))
    l_score = int(np.clip(rng.normal(65 + skill_l * 3, 10), 40, 110))
    if l_score >= w_score:
        l_score = w_score - rng.randint(1, 6)

    # Field goals
    w_fga = rng.randint(50, 75)
    w_fgm = int(w_fga * np.clip(rng.normal(0.44 + skill_w * 0.02, 0.05), 0.30, 0.60))
    l_fga = rng.randint(50, 75)
    l_fgm = int(l_fga * np.clip(rng.normal(0.40 + skill_l * 0.01, 0.05), 0.28, 0.55))

    # 3-pointers
    w_fga3 = rng.randint(15, 35)
    w_fgm3 = int(w_fga3 * np.clip(rng.normal(0.35, 0.06), 0.15, 0.55))
    l_fga3 = rng.randint(15, 35)
    l_fgm3 = int(l_fga3 * np.clip(rng.normal(0.33, 0.06), 0.15, 0.55))

    # Free throws
    w_fta = rng.randint(10, 30)
    w_ftm = int(w_fta * np.clip(rng.normal(0.72, 0.08), 0.50, 0.95))
    l_fta = rng.randint(10, 30)
    l_ftm = int(l_fta * np.clip(rng.normal(0.70, 0.08), 0.50, 0.95))

    return {
        "WScore": w_score, "LScore": l_score,
        "WFGM": w_fgm, "WFGA": w_fga, "WFGM3": w_fgm3, "WFGA3": w_fga3,
        "WFTM": w_ftm, "WFTA": w_fta,
        "WOR": rng.randint(7, 18), "WDR": rng.randint(18, 32),
        "WAst": rng.randint(10, 22), "WTO": rng.randint(8, 20),
        "WStl": rng.randint(3, 12), "WBlk": rng.randint(1, 8),
        "WPF": rng.randint(12, 25),
        "LFGM": l_fgm, "LFGA": l_fga, "LFGM3": l_fgm3, "LFGA3": l_fga3,
        "LFTM": l_ftm, "LFTA": l_fta,
        "LOR": rng.randint(7, 18), "LDR": rng.randint(18, 32),
        "LAst": rng.randint(8, 20), "LTO": rng.randint(8, 20),
        "LStl": rng.randint(3, 12), "LBlk": rng.randint(1, 8),
        "LPF": rng.randint(12, 25),
    }


def generate_regular_season_detailed():
    """Generate MRegularSeasonDetailedResults.csv."""
    rng = np.random.RandomState(SEED + 1)
    rows = []

    # Assign each team a skill level per season
    for season in SEASONS:
        team_ids = list(range(1100, 1100 + TOTAL_TEAMS))
        skills = {t: rng.normal(0, 1) for t in team_ids}

        # ~30 games per team, ~5500 games per season
        for day in range(0, 132):
            num_games = rng.randint(5, 25)
            for _ in range(num_games):
                t1, t2 = rng.choice(team_ids, size=2, replace=False)
                # Higher-skill team more likely to win
                p_t1_wins = 1 / (1 + np.exp(-(skills[t1] - skills[t2])))
                if rng.random() < p_t1_wins:
                    w_id, l_id = t1, t2
                else:
                    w_id, l_id = t2, t1

                stats = _generate_game_stats(rng, skills[w_id], skills[l_id])
                row = {"Season": season, "DayNum": day, "WTeamID": w_id,
                       "LTeamID": l_id, "WLoc": rng.choice(["H", "A", "N"]),
                       "NumOT": 0}
                row.update(stats)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv", index=False)
    return df


def generate_tourney_detailed(seeds_df):
    """Generate MNCAATourneyDetailedResults.csv."""
    rng = np.random.RandomState(SEED + 2)
    rows = []

    for season in SEASONS:
        season_seeds = seeds_df[seeds_df["Season"] == season]
        team_ids = season_seeds["TeamID"].values.tolist()
        seed_map = {}
        for _, r in season_seeds.iterrows():
            seed_str = r["Seed"]
            seed_num = int("".join(c for c in seed_str[1:] if c.isdigit()))
            seed_map[r["TeamID"]] = seed_num

        # Simulate a 64-team single-elimination tournament
        bracket = list(rng.choice(team_ids, size=min(64, len(team_ids)), replace=False))
        day = 134
        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                if i + 1 >= len(bracket):
                    next_round.append(bracket[i])
                    continue
                t1, t2 = bracket[i], bracket[i + 1]
                s1 = seed_map.get(t1, 8)
                s2 = seed_map.get(t2, 8)
                # Lower seed = better team, more likely to win
                skill_diff = (s2 - s1) * 0.15
                p_t1_wins = 1 / (1 + np.exp(-skill_diff))
                # Add some randomness for upsets
                p_t1_wins = 0.3 + 0.4 * p_t1_wins  # compress toward 50%

                if rng.random() < p_t1_wins:
                    w_id, l_id = t1, t2
                else:
                    w_id, l_id = t2, t1

                stats = _generate_game_stats(rng, -seed_map.get(w_id, 8) / 16,
                                             -seed_map.get(l_id, 8) / 16)
                row = {"Season": season, "DayNum": day, "WTeamID": w_id,
                       "LTeamID": l_id, "WLoc": "N", "NumOT": 0}
                row.update(stats)
                rows.append(row)
                next_round.append(w_id)
            bracket = next_round
            day += 2

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv", index=False)
    return df


def generate_massey_ordinals():
    """Generate MMasseyOrdinals.csv with KenPom and Sagarin rankings."""
    rng = np.random.RandomState(SEED + 3)
    rows = []
    systems = ["POM", "SAG"]  # KenPom and Sagarin

    for season in SEASONS:
        team_ids = list(range(1100, 1100 + TOTAL_TEAMS))
        # Generate true skill and rank
        true_skill = {t: rng.normal(0, 1) for t in team_ids}
        sorted_teams = sorted(team_ids, key=lambda t: -true_skill[t])

        for sys_name in systems:
            # Add noise to ranking
            noisy_skill = {t: true_skill[t] + rng.normal(0, 0.3)
                           for t in team_ids}
            ranked = sorted(team_ids, key=lambda t: -noisy_skill[t])
            rank_map = {t: i + 1 for i, t in enumerate(ranked)}

            # Only store the last rating day before tournament
            for t in team_ids:
                rows.append({
                    "Season": season,
                    "RankingDayNum": 133,
                    "SystemName": sys_name,
                    "TeamID": t,
                    "OrdinalRank": rank_map[t],
                })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "MMasseyOrdinals.csv", index=False)
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating teams...")
    generate_teams()
    print("Generating seeds...")
    seeds_df = generate_seeds()
    print("Generating regular season detailed results...")
    generate_regular_season_detailed()
    print("Generating tournament detailed results...")
    generate_tourney_detailed(seeds_df)
    print("Generating Massey ordinals...")
    generate_massey_ordinals()
    print(f"Done! Files written to {DATA_DIR}/")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {f.name}: {len(pd.read_csv(f))} rows")


if __name__ == "__main__":
    main()
