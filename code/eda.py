import glob
import os
import pandas as pd

# Store files in a list and sort them to ensure correct order
all_files = sorted(glob.glob("data/play_by_play_part*.csv"))

# Combine all the CSV files into a single dataframe
df = pd.concat(
    (pd.read_csv(f, low_memory=False) for f in all_files), ignore_index=True
)

# Clean scoremargin, replace TIE with 0, and forward-fill across each game
# so non-scoring plays like misses inherit the correct score margin from prior plays
df["scoremargin_clean"] = pd.to_numeric(
    df["scoremargin"].replace("TIE", 0), errors="coerce"
)
df["scoremargin_clean"] = df.groupby("game_id")["scoremargin_clean"].ffill()

# Split 'MM:SS' into minutes and seconds, then convert to total seconds remaining
time_parts = df["pctimestring"].str.split(":", expand=True)
minutes = pd.to_numeric(time_parts[0], errors="coerce")
seconds = pd.to_numeric(time_parts[1], errors="coerce")
total_seconds = (minutes * 60) + seconds

# Flag clutch rows instead of discarding non-clutch rows
df["is_clutch"] = (
    (df["period"] >= 4)
    & (total_seconds <= 300)
    & (df["scoremargin_clean"].abs() <= 5)
)

clutch_df = df[df["is_clutch"]]
print(f"Filtered down to {clutch_df.shape[0]} clutch rows!")

preview_cols = [
    "period",
    "pctimestring",
    "scoremargin",
    "player1_name",
    "homedescription",
    "visitordescription",
]
print(clutch_df[preview_cols].head(10))

# Isolate shooting events (Event 1 = made shot, Event 2 = missed shot)
shot_df = df[df["eventmsgtype"].isin([1, 2])].copy()
shot_df["shot_made"] = (shot_df["eventmsgtype"] == 1).astype(int)

clutch_shots = shot_df[shot_df["is_clutch"]]
non_clutch_shots = shot_df[~shot_df["is_clutch"]]

# Calculate non-clutch shooting stats per player (mutually exclusive baseline)
non_clutch_stats = (
    non_clutch_shots.groupby("player1_name")
    .agg(
        non_clutch_FGA=("shot_made", "count"),
        non_clutch_FGM=("shot_made", "sum"),
    )
    .reset_index()
)
non_clutch_stats["non_clutch_FG_pct"] = (
    non_clutch_stats["non_clutch_FGM"] / non_clutch_stats["non_clutch_FGA"]
)

# Calculate clutch shooting stats per player
clutch_stats = (
    clutch_shots.groupby("player1_name")
    .agg(
        clutch_FGA=("shot_made", "count"),
        clutch_FGM=("shot_made", "sum"),
    )
    .reset_index()
)
clutch_stats["clutch_FG_pct"] = (
    clutch_stats["clutch_FGM"] / clutch_stats["clutch_FGA"]
)

# Merge clutch and non-clutch stats (inner join keeps players with both)
comparison_df = pd.merge(
    non_clutch_stats, clutch_stats, on="player1_name", how="inner"
)
comparison_df["fg_diff"] = (
    comparison_df["clutch_FG_pct"] - comparison_df["non_clutch_FG_pct"]
)

# Filter for players with at least 25 clutch attempts
filtered_comparison = comparison_df[comparison_df["clutch_FGA"] >= 25].copy()

print("\n--- Top Clutch Risers (vs. Non-Clutch) ---")
print(
    filtered_comparison.sort_values(by="fg_diff", ascending=False)[
        [
            "player1_name",
            "non_clutch_FG_pct",
            "clutch_FG_pct",
            "fg_diff",
            "non_clutch_FGA",
            "clutch_FGA",
        ]
    ].head(5)
)

print("\n--- Biggest Efficiency Dips Under Pressure (vs. Non-Clutch) ---")
print(
    filtered_comparison.sort_values(by="fg_diff", ascending=True)[
        [
            "player1_name",
            "non_clutch_FG_pct",
            "clutch_FG_pct",
            "fg_diff",
            "non_clutch_FGA",
            "clutch_FGA",
        ]
    ].head(5)
)

os.makedirs("output", exist_ok=True)
comparison_df.to_csv("output/clutch_vs_non_clutch_all_players.csv", index=False)
filtered_comparison.to_csv("output/clutch_vs_non_clutch_qualified.csv", index=False)

print("\nSuccessfully exported CSV files to the 'output/' directory!")