import glob
import pandas as pd

# Store files in a list and sort them to ensure correct order
all_files = sorted(glob.glob("data/play_by_play_part*.csv"))

print(f"Found files: {all_files}")

# Combine all the CSV files into a single dataframe
df = pd.concat(
    (pd.read_csv(f, low_memory=False) for f in all_files), ignore_index=True
)

print(f"Successfully loaded dataset with {df.shape[0]} rows!")

# Filter for Clutch Situations
# Standard definition: 4th quarter or later (period >= 4), 
# score margin within 5 points, and 5 minutes or less remaining.

# Convert scoremargin to numeric safely (handling any non-numeric text rows like "TIE")
margin_numeric = pd.to_numeric(df['scoremargin'], errors='coerce')

clutch_df = df[
    (df['period'] >= 4) & 
    (margin_numeric.abs() <= 5)
]

print(f"Filtered down to {clutch_df.shape[0]} clutch rows!")

# Select key columns to inspect
preview_cols = ['period', 'pctimestring', 'scoremargin', 'player1_name', 'homedescription', 'visitordescription']
print(clutch_df[preview_cols].head(10))