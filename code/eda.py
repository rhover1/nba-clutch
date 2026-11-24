import glob
import os
import pandas as pd


def load_play_by_play(pattern: str = "data/play_by_play_part*.csv") -> pd.DataFrame:
    """Load and stack all play-by-play CSV parts."""
    files = sorted(glob.glob(pattern))
    return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)


def flag_clutch_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Clean score margin and clock, then add a boolean `is_clutch` column."""
    df = df.copy()

    # Replace TIE with 0 and forward-fill within each game for non-scoring plays
    margin = pd.to_numeric(df["scoremargin"].replace("TIE", 0), errors="coerce")
    df["scoremargin_clean"] = margin.groupby(df["game_id"]).ffill()

    # Convert 'MM:SS' clock to total seconds remaining in period
    time_parts = df["pctimestring"].str.split(":", expand=True)
    minutes = pd.to_numeric(time_parts[0], errors="coerce")
    seconds = pd.to_numeric(time_parts[1], errors="coerce")
    total_seconds = (minutes * 60) + seconds

    # Clutch: 4th quarter or OT, <= 5 mins remaining, score within 5 points
    df["is_clutch"] = (
        (df["period"] >= 4)
        & (total_seconds <= 300)
        & (df["scoremargin_clean"].abs() <= 5)
    )
    return df


def summarize_shots(shots: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate FGA, FGM, and FG% per player for a given subset of shots."""
    stats_df = (
        shots.groupby("player1_name")
        .agg(
            FGA=("shot_made", "count"),
            FGM=("shot_made", "sum"),
        )
        .reset_index()
    )
    stats_df["FG_pct"] = stats_df["FGM"] / stats_df["FGA"]
    return stats_df.rename(
        columns={
            "FGA": f"{prefix}_FGA",
            "FGM": f"{prefix}_FGM",
            "FG_pct": f"{prefix}_FG_pct",
        }
    )


def build_player_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare clutch vs. non-clutch FG% per player and return (all, qualified)."""
    shots = df[df["eventmsgtype"].isin([1, 2])].copy()
    shots["shot_made"] = (shots["eventmsgtype"] == 1).astype(int)

    non_clutch = summarize_shots(shots[~shots["is_clutch"]], "non_clutch")
    clutch = summarize_shots(shots[shots["is_clutch"]], "clutch")

    comparison = pd.merge(non_clutch, clutch, on="player1_name", how="inner")
    comparison["fg_diff"] = comparison["clutch_FG_pct"] - comparison["non_clutch_FG_pct"]

    qualified = comparison[comparison["clutch_FGA"] >= 25].copy()
    return comparison, qualified


def print_summary(df: pd.DataFrame, qualified: pd.DataFrame) -> None:
    """Print row attrition counts and top/bottom 5 clutch performers."""
    shots = df[df["eventmsgtype"].isin([1, 2])]
    print(f"Total raw rows: {len(df):,}")
    print(f"Total clutch rows: {df['is_clutch'].sum():,}")
    print(f"Total field goal attempts: {len(shots):,}")
    print(f"Clutch field goal attempts: {shots['is_clutch'].sum():,}")
    print(f"Qualified players (>= 25 clutch FGA): {len(qualified):,}")

    cols = [
        "player1_name",
        "non_clutch_FG_pct",
        "clutch_FG_pct",
        "fg_diff",
        "non_clutch_FGA",
        "clutch_FGA",
    ]
    print("\n--- Top 5 Clutch Risers ---")
    print(qualified.sort_values("fg_diff", ascending=False)[cols].head(5).to_string(index=False))

    print("\n--- Top 5 Efficiency Dips ---")
    print(qualified.sort_values("fg_diff", ascending=True)[cols].head(5).to_string(index=False))


def main() -> None:
    df = load_play_by_play()
    df = flag_clutch_plays(df)
    comparison_df, filtered_comparison = build_player_comparison(df)

    print_summary(df, filtered_comparison)

    os.makedirs("output", exist_ok=True)
    comparison_df.to_csv("output/clutch_vs_non_clutch_all_players.csv", index=False)
    filtered_comparison.to_csv("output/clutch_vs_non_clutch_qualified.csv", index=False)
    print("\nExported CSVs to 'output/'")


if __name__ == "__main__":
    main()