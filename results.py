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
    if song_results.empty:
        return
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
    if song_results.empty:
        print("df_results is empty; regression not possible.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}
    
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
    if song_results.empty:
        print("song_results is empty; skipping t_spec vs iters plot.")
        return

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
    if song_results.empty:
        print("song_results is empty; skipping histograms.")
        return

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