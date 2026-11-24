import pandas as pd
from scipy import stats


def load_qualified_data(
    path: str = "output/clutch_vs_non_clutch_qualified.csv",
) -> pd.DataFrame:
    """Load the qualified player clutch vs. non-clutch dataset."""
    return pd.read_csv(path)


def run_paired_tests(df: pd.DataFrame) -> dict:
    """Run paired t-test, Wilcoxon test, and 95% t-interval on within-player FG% diff."""
    clean = df.dropna(subset=["clutch_FG_pct", "non_clutch_FG_pct", "fg_diff"])
    diffs = clean["fg_diff"]
    n = len(diffs)

    t_stat, t_pval = stats.ttest_rel(
        clean["clutch_FG_pct"], clean["non_clutch_FG_pct"]
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
        f"Mean FG% diff: {res['mean_diff'] * 100:+.2f} pp "
        f"(95% CI: [{res['ci_low'] * 100:+.2f}, {res['ci_high'] * 100:+.2f}] pp)"
    )
    print(f"Paired t-test: t = {res['t_stat']:.3f}, p = {res['t_pval']:.4e}")


def main() -> None:
    qualified_df = load_qualified_data()
    results = run_paired_tests(qualified_df)
    print_test_results(results)


if __name__ == "__main__":
    main()