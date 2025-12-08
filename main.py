from billboard import BillboardDataset
from song_analysis import SongAnalysis

# Set the global configuration
SONG_DATA_PATH = "all_songs_data.csv"
YEAR_START = 1959
YEAR_END = 1991

MAX_SONGS_PER_YEAR = 20
MAX_CANDIDATES_PER_YEAR = 100
POWER_TOL = 1e-6
ALPHA = 1e-3
MIN_STATES = 3

# Load the billboard and song data
billboard_dataset = BillboardDataset()
billboard_dataset.build()

song_analysis = SongAnalysis(
    song_data_path=SONG_DATA_PATH,
    billboard_dataset=billboard_dataset,
    year_start=YEAR_START,
    year_end=YEAR_END,
    max_songs_per_year=MAX_SONGS_PER_YEAR,
    max_candidates_per_year=MAX_CANDIDATES_PER_YEAR,
    power_tol=POWER_TOL,
    alpha=ALPHA,
    min_states=MIN_STATES,
)

selected_songs = song_analysis.select_songs()