import glob
import os
import pandas as pd
import numpy as np

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
    """Compare clutch vs. non-clutch FG% and 2PT/3PT splits per player."""
    shots = df[df["eventmsgtype"].isin([1, 2])].copy()
    shots["shot_made"] = (shots["eventmsgtype"] == 1).astype(int)
    shots = tag_shot_types(shots)

    non_clutch = summarize_shots(shots[~shots["is_clutch"]], "non_clutch")
    clutch = summarize_shots(shots[shots["is_clutch"]], "clutch")

    comparison = pd.merge(non_clutch, clutch, on="player1_name", how="inner")
    comparison["fg_diff"] = comparison["clutch_FG_pct"] - comparison["non_clutch_FG_pct"]
    comparison = add_shot_mix_metrics(shots, comparison)

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

def tag_shot_types(shots: pd.DataFrame) -> pd.DataFrame:
    """Label field goal attempts (eventmsgtype 1 or 2) as 3PT or 2PT."""
    shots = shots.copy()
    home_desc = shots["homedescription"].fillna("")
    visitor_desc = shots["visitordescription"].fillna("")
    desc = (home_desc + " " + visitor_desc).str.upper()

    shots["is_3pt"] = desc.str.contains("3PT", regex=False)
    shots["shot_type"] = shots["is_3pt"].map({True: "3PT", False: "2PT"})
    return shots

def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide num by den safely, returning NaN where den == 0."""
    out = np.full(num.shape, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out

def add_shot_mix_metrics(
    shots: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Add 2PT/3PT attempt rates and FG% splits for non-clutch and clutch shots."""
    tagged = tag_shot_types(shots)

    def _player_split(subset: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grp = (
            subset.groupby(["player1_name", "is_3pt"])["shot_made"]
            .agg(att="count", made="sum")
            .unstack(fill_value=0)
        )
        for col in [False, True]:
            if ("att", col) not in grp.columns:
                grp[("att", col)] = 0
            if ("made", col) not in grp.columns:
                grp[("made", col)] = 0

        fga_2 = grp[("att", False)].to_numpy()
        fgm_2 = grp[("made", False)].to_numpy()
        fga_3 = grp[("att", True)].to_numpy()
        fgm_3 = grp[("made", True)].to_numpy()
        tot_fga = fga_2 + fga_3

        return pd.DataFrame(
            {
                "player1_name": grp.index.to_numpy(),
                f"{prefix}_2PA_rate": _safe_div(fga_2, tot_fga),
                f"{prefix}_3PA_rate": _safe_div(fga_3, tot_fga),
                f"{prefix}_2PT_FG_pct": _safe_div(fgm_2, fga_2),
                f"{prefix}_3PT_FG_pct": _safe_div(fgm_3, fga_3),
            }
        )

    nc_mix = _player_split(tagged[~tagged["is_clutch"]], "non_clutch")
    c_mix = _player_split(tagged[tagged["is_clutch"]], "clutch")

    out = comparison.merge(nc_mix, on="player1_name", how="left").merge(
        c_mix, on="player1_name", how="left"
    )
    out["3PA_rate_diff"] = out["clutch_3PA_rate"] - out["non_clutch_3PA_rate"]
    out["2PT_FG_diff"] = out["clutch_2PT_FG_pct"] - out["non_clutch_2PT_FG_pct"]
    out["3PT_FG_diff"] = out["clutch_3PT_FG_pct"] - out["non_clutch_3PT_FG_pct"]
    return out

def preview_shot_tags(df: pd.DataFrame, n: int = 10) -> None:
    """Print a quick sample of tagged 2PT/3PT shots to verify description parsing."""
    shots = tag_shot_types(df[df["eventmsgtype"].isin([1, 2])].copy())
    cols = ["homedescription", "visitordescription", "shot_type", "is_3pt"]
    print("\n--- Sample 2PT/3PT Tag Check ---")
    print(shots[cols].head(n).to_string(index=False))
    print("\nShot type counts across all FGA:")
    print(shots["shot_type"].value_counts())

def record_step(
    audit_log: list[dict],
    step_name: str,
    df: pd.DataFrame,
    prev_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append row, column, and missing-value changes after a pipeline step."""
    cols = set(df.columns)
    prev_cols = set(prev_df.columns) if prev_df is not None else cols

    added_cols = sorted(cols - prev_cols)
    dropped_cols = sorted(prev_cols - cols)
    rows_delta = len(df) - len(prev_df) if prev_df is not None else 0
    na_delta = int(df.isna().sum().sum() - prev_df.isna().sum().sum()) if prev_df is not None else 0

    audit_log.append({
        "step": step_name,
        "rows": len(df),
        "rows_delta": rows_delta,
        "cols": len(df.columns),
        "added_cols": ", ".join(added_cols) if added_cols else "-",
        "dropped_cols": ", ".join(dropped_cols) if dropped_cols else "-",
        "na_cells_delta": na_delta,
    })
    return df


def print_audit_log(audit_log: list[dict]) -> pd.DataFrame:
    """Print summary counts and return the full audit DataFrame."""
    log_df = pd.DataFrame(audit_log)
    summary_cols = ["step", "rows", "rows_delta", "cols", "na_cells_delta"]
    print("\n--- Pipeline Step Audit ---")
    print(log_df[summary_cols].to_string(index=False))
    return log_df

def main() -> None:
    audit_log: list[dict] = []

    raw_df = load_play_by_play()
    record_step(audit_log, "1. Load raw play-by-play CSVs", raw_df)

    flagged_df = flag_clutch_plays(raw_df)
    record_step(audit_log, "2. Clean margin/clock & flag clutch", flagged_df, raw_df)

    shots_df = flagged_df[flagged_df["eventmsgtype"].isin([1, 2])].copy()
    shots_df["shot_made"] = (shots_df["eventmsgtype"] == 1).astype(int)
    shots_df = tag_shot_types(shots_df)
    record_step(audit_log, "3. Filter FGA (events 1,2) & tag 2PT/3PT", shots_df, flagged_df)

    comparison_df, filtered_comparison = build_player_comparison(flagged_df)
    record_step(audit_log, "4. Player comparison (all paired)", comparison_df, shots_df)
    record_step(audit_log, "5. Qualified players (>= 25 clutch FGA)", filtered_comparison, comparison_df)

    print_summary(flagged_df, filtered_comparison)
    log_df = print_audit_log(audit_log)

    os.makedirs("output", exist_ok=True)
    comparison_df.to_csv("output/clutch_vs_non_clutch_all_players.csv", index=False)
    filtered_comparison.to_csv("output/clutch_vs_non_clutch_qualified.csv", index=False)
    log_df.to_csv("output/pipeline_step_log.csv", index=False)
    print("\nExported CSVs to 'output/'")


if __name__ == "__main__":
    main()
