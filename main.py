"""
Runs the song analysis pipeline.
"""

import pandas as pd
from billboard import BillboardDataset
from song_analysis import SongAnalysis
from results import (
    print_small_gap_songs, 
    regression_tspec_vs_iters,
    plot_tspec_vs_iters_loglog,
    plot_histograms,
)

# Set the global configuration
SONG_DATA_PATH = "all_songs_data.csv"
YEAR_START = 1959
YEAR_END = 1991

MAX_SONGS_PER_YEAR = 20
MAX_CANDIDATES_PER_YEAR = 100
POWER_METHOD_TOL = 1e-6
ALPHA = 1e-3
MIN_STATES = 3

GATHER_DATA = False

# Gather the song results
if GATHER_DATA:
    # Load the billboard and song data
    billboard_dataset = BillboardDataset()
    billboard_dataset.build()

    # Analyze the songs
    song_analysis = SongAnalysis(
        song_data_path=SONG_DATA_PATH,
        billboard_dataset=billboard_dataset,
        year_start=YEAR_START,
        year_end=YEAR_END,
        max_songs_per_year=MAX_SONGS_PER_YEAR,
        max_candidates_per_year=MAX_CANDIDATES_PER_YEAR,
        power_method_tol=POWER_METHOD_TOL,
        alpha=ALPHA,
        min_states=MIN_STATES,
    )
    selected_songs = song_analysis.select_songs()
    song_results = song_analysis.analyze_songs(selected_songs)
    if song_results.empty:
        raise ValueError("No songs were successfully analyzed.")

    # Create the results plots from the song analysis
    song_results.to_csv("song_results.csv", index=False)
else:
    song_results = pd.read_csv("song_results.csv")

# Run analysis on the song results

# Print some of the most problematic tiny-gap songs
print_small_gap_songs(song_results, k=10)
# Can t_spec predict t_PM (on the reasonable subset)?
stats_all = regression_tspec_vs_iters(
    song_results, filter_converged=False, max_tspec=None
)
print()
print("\nRegression on all songs (including non-converged and huge t_spec):")
print(stats_all)

stats_filtered = regression_tspec_vs_iters(
    song_results, filter_converged=True, max_tspec=1e4
)
print()
print("Regression on converged songs with t_spec <= 1e4:")
print(stats_filtered)

# Visualize the results
plot_tspec_vs_iters_loglog(song_results)
plot_histograms(song_results)
# plot_gap_vs_states(song_results)