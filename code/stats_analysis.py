import pandas as pd
from scipy import stats


def load_qualified_data(
    path: str = "output/clutch_vs_non_clutch_qualified.csv",
) -> pd.DataFrame:
    """Load the qualified player clutch vs. non-clutch dataset."""
    return pd.read_csv(path)


def run_paired_tests(df: pd.DataFrame, 
                     clutch_col: str,
                     non_clutch_col: str,
                     diff_col: str,
                    ) -> dict:
    """Run a paired t-test and compute a 95% confidence interval for within-player differences."""
    clean = df.dropna(subset=[clutch_col, non_clutch_col, diff_col])
    diffs = clean[diff_col]
    n = len(diffs)

    t_stat, t_pval = stats.ttest_rel(
        clean[clutch_col], clean[non_clutch_col]
    )

    mean_diff = diffs.mean()
    se_diff = stats.sem(diffs)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_pval": t_pval,
    }


def print_test_results(res: dict) -> None:
    """Format and print hypothesis test and confidence interval results."""
    print(f"Qualified players (n): {res['n']}")
    print(
        f"Mean difference: {res['mean_diff'] * 100:+.2f} pp "
        f"(95% CI: [{res['ci_low'] * 100:+.2f}, {res['ci_high'] * 100:+.2f}] pp)"
    )
    print(f"Paired t-test: t = {res['t_stat']:.3f}, p = {res['t_pval']:.4e}")


def main() -> None:
    qualified_df = load_qualified_data()
    results_overall = run_paired_tests(qualified_df, "clutch_FG_pct", "non_clutch_FG_pct", "fg_diff")
    results_2PT = run_paired_tests(qualified_df, "clutch_2PT_FG_pct", "non_clutch_2PT_FG_pct", "2PT_FG_diff")
    results_3PT = run_paired_tests(qualified_df, "clutch_3PT_FG_pct", "non_clutch_3PT_FG_pct", "3PT_FG_diff")
    results_3PA_rate = run_paired_tests(qualified_df, "clutch_3PA_rate", "non_clutch_3PA_rate", "3PA_rate_diff")

    print("\n--- Overall FG% ---")
    print_test_results(results_overall)
    print("\n--- 2PT FG% ---")
    print_test_results(results_2PT)
    print("\n--- 3PT FG% ---")
    print_test_results(results_3PT)
    print("\n--- 3PA Rate% ---")
    print_test_results(results_3PA_rate)


if __name__ == "__main__":
    main()