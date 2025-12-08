"""
Utility functions for analyzing and plotting the song analysis results.
"""
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

