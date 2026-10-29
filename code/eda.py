import glob
import pandas as pd
import os

# Store files in a list and sort them to ensure correct order
all_files = sorted(glob.glob("data/play_by_play_part*.csv"))

# Combine all the CSV files into a single dataframe
df = pd.concat(
    (pd.read_csv(f, low_memory=False) for f in all_files), ignore_index=True
)

# Filter for Clutch Situations
# Standard definition: 4th quarter or later (period >= 4), 
# score margin within 5 points, and 5 minutes or less remaining.

# Clean scoremargin, replace TIE with 0, and forward-fill across each game 
# so non-scoring plays like misses inherit the correct score margin from prior plays
df['scoremargin_clean'] = pd.to_numeric(
    df['scoremargin'].replace('TIE', 0), errors='coerce'
)
df['scoremargin_clean'] = df.groupby('game_id')['scoremargin_clean'].ffill()

# Convert scoremargin to numeric safely (handling any non-numeric text rows like "TIE")
margin_numeric = df['scoremargin_clean']

# Split 'MM:SS' into minutes and seconds, then convert to total seconds remaining
time_parts = df['pctimestring'].str.split(':', expand=True)
minutes = pd.to_numeric(time_parts[0], errors='coerce')
seconds = pd.to_numeric(time_parts[1], errors='coerce')
total_seconds = (minutes * 60) + seconds

clutch_df = df[
    (df['period'] >= 4) & 
    (total_seconds <= 300) & 
    (margin_numeric.abs() <= 5)
]

print(f"Filtered down to {clutch_df.shape[0]} clutch rows!")

# Select key columns to inspect
preview_cols = ['period', 'pctimestring', 'scoremargin', 'player1_name', 'homedescription', 'visitordescription']
print(clutch_df[preview_cols].head(10))

# Isolate shooting events (Event 1 = made shot, Event 2 = missed shot)
shot_df = df[df['eventmsgtype'].isin([1, 2])].copy()
shot_df['shot_made'] = (shot_df['eventmsgtype'] == 1).astype(int)

clutch_shots = clutch_df[clutch_df['eventmsgtype'].isin([1, 2])].copy()
clutch_shots['shot_made'] = (clutch_shots['eventmsgtype'] == 1).astype(int)

# Calculate overall season shooting stats per player
overall_stats = shot_df.groupby('player1_name').agg(
    overall_FGA=('shot_made', 'count'),
    overall_FGM=('shot_made', 'sum')
).reset_index()
overall_stats['overall_FG_pct'] = overall_stats['overall_FGM'] / overall_stats['overall_FGA']

# Calculate clutch shooting stats per player
clutch_stats = clutch_shots.groupby('player1_name').agg(
    clutch_FGA=('shot_made', 'count'),
    clutch_FGM=('shot_made', 'sum')
).reset_index()
clutch_stats['clutch_FG_pct'] = clutch_stats['clutch_FGM'] / clutch_stats['clutch_FGA']

# Merge together and filter for players with a minimum number of clutch attempts to ensure statistical significance
comparison_df = pd.merge(overall_stats, clutch_stats, on='player1_name')
filtered_comparison = comparison_df[comparison_df['clutch_FGA'] >= 25].copy()

# Calculate the efficiency difference (Clutch FG% minus Overall FG%)
filtered_comparison['fg_diff'] = filtered_comparison['clutch_FG_pct'] - filtered_comparison['overall_FG_pct']

# Sort to see who rises vs drops the most under pressure
print("\n--- Top Clutch Risers ---")
print(filtered_comparison.sort_values(by='fg_diff', ascending=False)[['player1_name', 'overall_FG_pct', 'clutch_FG_pct', 'clutch_FGA']].head(5))

print("\n--- Biggest Efficiency Dips Under Pressure ---")
print(filtered_comparison.sort_values(by='fg_diff', ascending=True)[['player1_name', 'overall_FG_pct', 'clutch_FG_pct', 'clutch_FGA']].head(5))

print("\n--- Comparison DataFrame (All Players) ---")
print(comparison_df.head(10))

print("\n--- Filtered Comparison DataFrame (Players with >= 25 Clutch Attempts) ---")
print(filtered_comparison.head(10))

# Create an output directory
os.makedirs("output", exist_ok=True)

# Export DataFrames to CSV files
comparison_df.to_csv("output/clutch_vs_overall_all_players.csv", index=False)
filtered_comparison.to_csv("output/clutch_vs_overall_qualified.csv", index=False)

print("\nSuccessfully exported CSV files to the 'output/' directory!")