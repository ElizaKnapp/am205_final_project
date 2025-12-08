"""
Utility functions for analyzing and plotting the song analysis results.
"""
import numpy as np
import pandas as pd

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
        f"Regression (filtered={filter_converged}, max_tspec={max_tspec}): t_spec ~ {slope:.4f} * power_iters + {intercept:.4f} (R^2={r2:.3f}).",
    )
    return {"slope": slope, "intercept": intercept, "R2": r2}