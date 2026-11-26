import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_qualified_csv(path: str="output/clutch_vs_non_clutch_qualified.csv",)-> pd.DataFrame:
    return pd.read_csv(path)

def plot_histogram(df: pd.DataFrame, column: str, title: str, filename: str, bins: int=30,)->None:
    #plot distributions of percentage differences (clutch FG% - non clutch FG%)
    clean=df[column].dropna()*100  # to have percentage points
    
    n=len(clean)
    mean_val=clean.mean()
    median_val=clean.median()

    worsened = (clean < 0).sum()
    improved = (clean > 0).sum()
    unchanged = (clean == 0).sum()
    worsened_percentage=(worsened/n)*100
    improved_percentage=(improved/n)*100


    plt.figure(figsize=(8,5))
    plt.hist(clean, bins=np.arange(-100, 102, 2), edgecolor="black")
    plt.axvline(0, linestyle="--", linewidth=1.5, label="No change", color="black")
    plt.axvline(mean_val, linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.2f} pp", color="red")
    plt.xlim(-50, 50)
    plt.ylim(0, 200)

    plt.title(title)
    plt.xlabel("Clutch - Non-clutch FG% difference (percentage points)")
    plt.ylabel("Number of players")
    plt.legend()

    summary_text = (
        f"n = {n}\n"
        f"Worsened = {worsened} ({worsened/n:.1%})\n"
        f"Improved = {improved} ({improved/n:.1%})\n"
        f"Unchanged = {unchanged} ({unchanged/n:.1%})\n"
        f"Mean = {mean_val:.2f} pp\n"
        f"Median = {median_val:.2f} pp\n"
    )

    plt.text(
        0.98,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300)
    plt.close()

def main() -> None:
    df=load_qualified_csv()

    plot_histogram(
        df,
        column="fg_diff",
        title="Distribution of Overall FG% Changes in Clutch Time",
        filename="hist_overall_fg_diff.png",
    )
    
    plot_histogram(
        df,
        column="2PT_FG_diff",
        title="Distribution of 2PT FG% Changes in Clutch Time",
        filename="hist_2pt_fg_diff.png",
    )

    plot_histogram(
        df,
        column="3PT_FG_diff",
        title="Distribution of 3PT FG% Changes in Clutch Time",
        filename="hist_3pt_fg_diff.png",
    )

    print("Histogram figures have been saved to 'figures/'")


if __name__ == "__main__":
    main()