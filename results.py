"""
Utility functions for analyzing and plotting the song analysis results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_small_gap_songs(song_results: pd.DataFrame, k: int = 10):
    """
    Print the top-k songs with the smallest spectral gaps for inspection.
    """
    df_sorted = song_results.sort_values("spectral_gap").head(k)
    cols = [
        "year",
        "decade",
        "artist",
        "title",
        "spectral_gap",
        "t_spec",
        "power_iters",
        "converged",
    ]
    print("\nSongs with the smallest spectral gaps (possible problematic cases):")
    print(df_sorted[cols].to_string(index=False))

def regression_tspec_vs_iters(song_results: pd.DataFrame, filter_converged: bool = True, max_tspec: float = None):
    """
    Simple linear regression: t_PM ~ a * t_spec + b.
    """
    # Filter the results
    df = song_results.copy()
    if filter_converged and "converged" in df.columns:
        df = df[df["converged"]]
    if max_tspec is not None:
        df = df[df["t_spec"] <= max_tspec]

    if df.empty:
        print("No rows left after regression filtering.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}
    
    x = df["t_spec"].values
    y = df["power_iters"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        print("Not enough finite points for regression.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}
    
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(
        f"Regression (filtered={filter_converged}, max_tspec={max_tspec}): power_iters ~ {slope:.4f} * t_spec + {intercept:.4f} (R^2={r2:.3f}).",
    )
    return {"slope": slope, "intercept": intercept, "R2": r2}

def plot_tspec_vs_iters_loglog(song_results: pd.DataFrame):
    """
    Scatter plot (log–log) of t_spec vs t_PM, color-coded by number of states,
    and with marker shapes indicating convergence (circle = converged, x = not).
    """
    # Filter positive values for log scale
    df = song_results[
        (song_results["t_spec"] > 0) &
        (song_results["power_iters"] > 0) &
        (song_results["n_states"] > 0)
    ].copy()

    if df.empty:
        print("No positive t_spec / power_iters values to plot.")
        return

    # Prepare data
    x = np.log10(df["t_spec"])
    y = np.log10(df["power_iters"])
    states = df["n_states"]

    plt.figure(figsize=(7, 5))

    # Converged points: circles
    sc1 = plt.scatter(
        x, y,
        c=states,
        cmap="viridis",
        alpha=0.75,
        s=45,
        label="Converged",
        edgecolors="k"
    )

    # Colorbar shows number of states
    cbar = plt.colorbar(sc1)
    cbar.set_label("Number of chord states (n_states)", fontsize=10)

    plt.xlabel("log10(t_spec)", fontsize=12)
    plt.ylabel("log10(t_PM)", fontsize=12)
    plt.title("Spectral Mixing-Time Estimate vs Power-Method Iterations\nColor = number of chord states", fontsize=13)

    plt.legend()
    plt.tight_layout()
    plt.savefig("spectral_mixing_vs_power_iterations.png", dpi=150)
    plt.close()

def plot_histograms(song_results: pd.DataFrame):
    """
    Plot histograms of spectral gaps and power-method iterations.
    """
    # Histogram of the spectral gap values for each song
    plt.figure(figsize=(6, 4))
    plt.hist(song_results["spectral_gap"].values, bins=30, alpha=0.8)
    plt.xlabel("Spectral gap")
    plt.ylabel("Count of songs")
    plt.title("Distribution of spectral gaps across songs")
    plt.tight_layout()
    plt.savefig("spectral_gap_distribution.png", dpi=150)
    plt.close()

    # Histogram of the power iteration taken for each song
    plt.figure(figsize=(6, 4))
    plt.hist(song_results["power_iters"].values, bins=30, alpha=0.8)
    plt.xlabel("Power-method iterations")
    plt.ylabel("Count of songs")
    plt.title("Distribution of power-method iterations")
    plt.tight_layout()
    plt.savefig("power_iterations_distribution.png", dpi=150)
    plt.close()

def plot_gap_vs_states(song_results: pd.DataFrame):
    """
    Scatter plot of spectral gap vs number of states to see if larger
    state spaces tend to have smaller gaps.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(song_results["n_states"], song_results["spectral_gap"], alpha=0.7)
    plt.xlabel("Number of chord states in song")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap vs number of chord states")
    plt.tight_layout()
    plt.savefig("spectral_gap_vs_states.png", dpi=150)
    plt.close()

def aggregate_by_year_from_song_results(song_results: pd.DataFrame):
    """
    Aggregate the song results by year.
    """
    grp = song_results.groupby("year")
    df_year = grp.agg(
        median_gap=("spectral_gap", "median"),
        median_iters=("power_iters", "median"),
    ).reset_index()
    return df_year

def plot_year_trends(aggregate_song_results: pd.DataFrame):
    """
    Plot trends in median spectral gap and median power-method iterations over years.
    Creates three graphs: full range, 1960-1975, and 1975-1990.
    """
    df = aggregate_song_results.copy()
    
    def create_plot(df_subset, title_suffix, filename):
        if df_subset.empty:
            print(f"No data for {title_suffix}")
            return
        
        years = df_subset["year"].values
        median_gap = df_subset["median_gap"].values
        median_iters = df_subset["median_iters"].values

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Median spectral gap",  alpha=0.8, color="blue")
        ax1.plot(years, median_gap, marker="o", alpha=0.8, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Median power-method iterations", alpha=0.8, color="red")
        ax2.plot(years, median_iters, marker="s", linestyle="--", alpha=0.8, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        
        plt.title(f"Year-level trends in median spectral gap and median power-method iterations ({title_suffix})")
        fig.tight_layout()
        plt.savefig(filename, dpi=150) 
        plt.close()
    
    # All years plot
    create_plot(df, "all years", "year_trends.png")
    # Split graphs to normalize scales
    df_early = df[(df["year"] >= 1960) & (df["year"] <= 1975)]
    df_late = df[(df["year"] >= 1975) & (df["year"] <= 1990)]
    create_plot(df_early, "1960-1975", "year_trends_1960_1975.png")
    create_plot(df_late, "1975-1990", "year_trends_1975_1990.png")