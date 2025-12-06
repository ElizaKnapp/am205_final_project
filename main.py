"""
AM205 Project: Spectral Mixing Times in Chord-Transition Markov Chains
(Updated for 1959–1991, more songs/year, diagnostics & plots)

Now excludes songs with fewer than 3 distinct chord states (n_states < 3),
since 2-state chains tend to produce pathological tiny spectral gaps and
blow up the t_spec bound.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mirdata  # for McGill Billboard dataset

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility Functions: normalization, filenames, etc.
# ---------------------------------------------------------------------------

def normalize_string_for_match(s: str) -> str:
    """
    Normalize a string for matching: lowercase, remove non-alphanumerics.
    """
    s = str(s).lower()
    return "".join(ch for ch in s if ch.isalnum())


def sanitize_for_filename(s: str) -> str:
    """
    Sanitize a string for use in filenames: lowercase, keep alnum + few safe chars.
    """
    s = str(s).lower()
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))


def make_song_key(year: int, artist: str, title: str) -> str:
    """
    Construct a reproducible key for a song for checkpoint filenames.
    """
    return f"{year}_{sanitize_for_filename(artist)}_{sanitize_for_filename(title)}"


# ---------------------------------------------------------------------------
# 1. Load Kaggle dataset
# ---------------------------------------------------------------------------

def load_kaggle_top_songs(csv_path: str) -> pd.DataFrame:
    """
    Load the Kaggle dataset with columns:

    ['Album', 'Album URL', 'Artist', 'Featured Artists', 'Lyrics', 'Media',
     'Rank', 'Release Date', 'Song Title', 'Song URL', 'Writers', 'Year']

    and normalize key columns to: 'year','rank','title','artist'.
    """
    logger.info("Loading Kaggle dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "Year": "year",
            "Rank": "rank",
            "Song Title": "title",
            "Artist": "artist",
        }
    )

    for col in ["year", "rank", "title", "artist"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in Kaggle file. "
                f"Available columns: {df.columns}"
            )

    # Coerce types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    # Drop rows with missing critical info
    before = len(df)
    df = df.dropna(subset=["year", "rank", "title", "artist"])
    after = len(df)
    if after < before:
        logger.warning(
            "Dropped %d rows from Kaggle dataset due to missing year/rank/title/artist.",
            before - after,
        )

    logger.info("Loaded Kaggle dataset with %d rows after cleaning.", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. McGill Billboard dataset loader & track lookup
# ---------------------------------------------------------------------------

def initialize_billboard_dataset(data_home: Optional[str] = None):
    """
    Initialize and (attempt to) download the McGill Billboard dataset via mirdata.

    If the download fails, log the error and try to continue with whatever
    is already on disk. If loading tracks still fails, return an empty dict.
    """
    logger.info("Initializing McGill Billboard dataset via mirdata.")
    billboard = mirdata.initialize("billboard", data_home=data_home)

    try:
        billboard.download()
    except Exception as exc:
        logger.error(
            "Billboard download failed (%s). "
            "Continuing with whatever data is already present on disk. "
            "Some songs may be skipped.",
            exc,
        )

    try:
        tracks = billboard.load_tracks()
        logger.info("Loaded Billboard dataset with %d tracks.", len(tracks))
    except Exception as exc:
        logger.error(
            "Failed to load Billboard tracks after download attempt: %s. "
            "Proceeding with no tracks; all songs will be skipped.",
            exc,
        )
        tracks = {}

    return billboard, tracks


def build_billboard_lookup(
    tracks: Dict[str, "mirdata.datasets.billboard.Track"],
) -> Dict[Tuple[str, str], List[str]]:
    """
    Build a map from (normalized_title, normalized_artist) -> list of track_ids.
    """
    logger.info("Building Billboard lookup by (title, artist).")
    lookup: Dict[Tuple[str, str], List[str]] = {}
    for track_id, track in tracks.items():
        title_norm = normalize_string_for_match(getattr(track, "title", "") or "")
        artist_norm = normalize_string_for_match(getattr(track, "artist", "") or "")
        key = (title_norm, artist_norm)
        lookup.setdefault(key, []).append(track_id)
    logger.info("Billboard lookup built with %d unique (title, artist) keys.", len(lookup))
    return lookup


def match_kaggle_row_to_billboard_track(
    row: pd.Series,
    billboard_lookup: Dict[Tuple[str, str], List[str]],
) -> Optional[str]:
    """
    Attempt to match a Kaggle row to a billboard track_id using normalized
    (title, artist). If multiple matches, just return the first.
    If no match, return None.
    """
    title_norm = normalize_string_for_match(row["title"])
    artist_norm = normalize_string_for_match(row["artist"])
    key = (title_norm, artist_norm)

    track_ids = billboard_lookup.get(key, [])
    if not track_ids:
        return None
    return track_ids[0]


# ---------------------------------------------------------------------------
# 3. Chord parsing & key-normalized roman numerals
# ---------------------------------------------------------------------------

# Pitch class mapping for chord roots
PC_FROM_NAME = {
    "c": 0, "b#": 0,
    "c#": 1, "db": 1,
    "d": 2,
    "d#": 3, "eb": 3,
    "e": 4, "fb": 4,
    "f": 5, "e#": 5,
    "f#": 6, "gb": 6,
    "g": 7,
    "g#": 8, "ab": 8,
    "a": 9,
    "a#": 10, "bb": 10,
    "b": 11, "cb": 11,
}


def parse_harte_chord_label(label: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a chord label in Harte/MIREX style like 'C:maj', 'G:min7', 'F#', 'N', etc.

    Returns
    -------
    root_pc : int or None
        Pitch class (0=C, 1=C#, ..., 11=B), or None if no chord.
    quality_simple : str or None
        One of {'maj','min','other'}, or None if no chord.
    """
    label = str(label).strip()
    if label in {"N", "X", ""}:
        return None, None

    if ":" in label:
        root_str, rest = label.split(":", 1)
    else:
        root_str, rest = label, ""

    root_str = root_str.lower()
    root_pc = PC_FROM_NAME.get(root_str)
    if root_pc is None:
        return None, None

    rest = rest.lower()
    if rest.startswith("maj"):
        quality = "maj"
    elif rest.startswith("min"):
        quality = "min"
    else:
        quality = "other"

    return root_pc, quality


def parse_tonic_string(tonic: str) -> Tuple[Optional[int], str]:
    """
    Parse tonic string from McGill SALAMI metadata, e.g. 'C:maj', 'a:min'.

    Returns
    -------
    tonic_pc : int or None
    mode : str
        'major' or 'minor' (default 'major' if ambiguous).
    """
    tonic = str(tonic).strip().lower()
    if ":" in tonic:
        root_str, rest = tonic.split(":", 1)
    else:
        root_str, rest = tonic, ""

    tonic_pc = PC_FROM_NAME.get(root_str)
    if tonic_pc is None:
        return None, "major"

    rest = rest.lower()
    if rest.startswith("min"):
        mode = "minor"
    else:
        mode = "major"
    return tonic_pc, mode


def roman_numeral_for_chord(
    root_pc: int,
    quality: str,
    tonic_pc: int,
    mode: str,
) -> str:
    """
    Map chord (root_pc, quality) to a roman numeral relative to tonic_pc.
    Very simple diatonic mapping; everything non-diatonic => 'OTHER'.
    """
    degree_pc = (root_pc - tonic_pc) % 12

    if mode == "major":
        mapping_major = {
            (0, "maj"): "I",
            (0, "min"): "i",
            (2, "min"): "ii",
            (2, "maj"): "II",
            (4, "min"): "iii",
            (4, "maj"): "III",
            (5, "maj"): "IV",
            (5, "min"): "iv",
            (7, "maj"): "V",
            (7, "min"): "v",
            (9, "min"): "vi",
            (9, "maj"): "VI",
            (11, "maj"): "VII",
            (11, "min"): "vii",
        }
        label = mapping_major.get((degree_pc, quality), "OTHER")
    else:
        mapping_minor = {
            (0, "min"): "i",
            (0, "maj"): "I",
            (2, "min"): "ii",
            (3, "maj"): "III",
            (5, "min"): "iv",
            (5, "maj"): "IV",
            (7, "min"): "v",
            (7, "maj"): "V",
            (8, "maj"): "VI",
            (10, "maj"): "VII",
        }
        label = mapping_minor.get((degree_pc, quality), "OTHER")

    return label


def simplify_roman_sequence(roman_chords: List[str]) -> List[str]:
    """
    Simplify roman-numeral chord sequence by compressing consecutive duplicates.
    """
    if not roman_chords:
        return []
    seq = [roman_chords[0]]
    for ch in roman_chords[1:]:
        if ch != seq[-1]:
            seq.append(ch)
    return seq


def get_chords_for_billboard_track(track) -> List[str]:
    """
    Extract a key-normalized roman-numeral chord sequence for a McGill
    Billboard track using its chord annotations and tonic metadata.
    """
    chord_data = getattr(track, "chords_majmin", None)
    if chord_data is None or getattr(chord_data, "labels", None) is None:
        logger.debug(
            "Track %s has no chords_majmin labels; skipping.",
            getattr(track, "track_id", "UNKNOWN"),
        )
        return []

    labels = chord_data.labels

    salami_metadata = getattr(track, "salami_metadata", None)
    if not salami_metadata or "tonic" not in salami_metadata:
        logger.debug(
            "Track %s has no tonic metadata; skipping.",
            getattr(track, "track_id", "UNKNOWN"),
        )
        return []

    tonic_info = salami_metadata["tonic"]
    tonic_pc, mode = parse_tonic_string(tonic_info)
    if tonic_pc is None:
        logger.debug(
            "Track %s tonic %r not understood; skipping.",
            getattr(track, "track_id", "UNKNOWN"),
            tonic_info,
        )
        return []

    roman_frames: List[str] = []
    for lab in labels:
        root_pc, q = parse_harte_chord_label(lab)
        if root_pc is None or q is None:
            continue
        rn = roman_numeral_for_chord(root_pc, q, tonic_pc, mode)
        roman_frames.append(rn)

    roman_seq = simplify_roman_sequence(roman_frames)
    return roman_seq


# ---------------------------------------------------------------------------
# 4. Markov chain: transition matrix, stationary distribution, power method
# ---------------------------------------------------------------------------

def build_transition_matrix_from_chords(
    chords: List[str],
    alpha: float = 1e-3,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build row-stochastic transition matrix P (with smoothing) from a
    key-normalized chord sequence.

    States are unique chords in order of first appearance.
    """
    if len(chords) < 2:
        raise ValueError("Chord sequence must have at least 2 chords to define transitions.")

    states: List[str] = []
    idx: Dict[str, int] = {}
    for ch in chords:
        if ch not in idx:
            idx[ch] = len(states)
            states.append(ch)

    n = len(states)
    counts = np.zeros((n, n), dtype=float)

    for c_from, c_to in zip(chords[:-1], chords[1:]):
        i = idx[c_from]
        j = idx[c_to]
        counts[i, j] += 1.0

    P = counts + alpha
    P = P / P.sum(axis=1, keepdims=True)
    return P, states


def compute_stationary_distribution_linear(P: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution pi by solving (P^T - I) pi = 0
    with sum(pi) = 1.
    """
    n = P.shape[0]
    PT = P.T
    A = PT - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    return pi


def power_method_to_stationary(
    P: np.ndarray,
    pi_exact: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iter: int = 1_000_000,
) -> Tuple[np.ndarray, int, bool]:
    """
    Run power method mu_{k+1}^T = mu_k^T P.

    If pi_exact is provided, stop when ||mu_k - pi_exact||_1 < tol.
    Otherwise, stop when ||mu_k - mu_{k-1}||_1 < tol.

    Returns
    -------
    mu : np.ndarray
        Final distribution.
    iters : int
        Number of iterations run.
    converged : bool
        True if the tolerance was met before hitting max_iter.
    """
    n = P.shape[0]
    mu = np.ones(n) / n
    prev = mu.copy()

    for k in range(1, max_iter + 1):
        mu = mu @ P
        if pi_exact is not None:
            dist = np.linalg.norm(mu - pi_exact, ord=1)
        else:
            dist = np.linalg.norm(mu - prev, ord=1)
        if dist < tol:
            return mu, k, True
        prev = mu.copy()

    logger.warning(
        "Power method did not converge within %d iterations (final dist=%.3e).",
        max_iter,
        dist,
    )
    return mu, max_iter, False


def compute_spectral_gap(P: np.ndarray) -> Tuple[float, float]:
    """
    Compute |lambda_2| and spectral gap gamma = 1 - |lambda_2|.
    """
    evals, _ = np.linalg.eig(P.T)
    evals = np.asarray(evals)
    idx = np.argsort(-np.abs(evals))
    evals_sorted = evals[idx]
    if len(evals_sorted) < 2:
        return 0.0, 1.0
    lambda2 = evals_sorted[1]
    lambda2_abs = float(np.abs(lambda2))
    gap = 1.0 - lambda2_abs
    return lambda2_abs, gap


def spectral_mixing_time_estimate(
    gap: float,
    tol: float,
    a2_estimate: float = 1.0,
) -> float:
    """
    Spectral mixing-time estimate t_spec ~ log(a2 / tol) / gap.

    This is a loose upper-bound-style approximation that is particularly
    fragile when gap is very small.
    """
    if gap <= 0:
        return float("inf")
    C = np.log(a2_estimate / tol)
    return float(C / gap)


# ---------------------------------------------------------------------------
# 5. Song selection based on year range & chord availability
# ---------------------------------------------------------------------------

def select_songs_year_range(
    df: pd.DataFrame,
    tracks: Dict[str, "mirdata.datasets.billboard.Track"],
    billboard_lookup: Dict[Tuple[str, str], List[str]],
    start_year: int = 1959,
    end_year: int = 1991,
    max_songs_per_year: int = 20,
    max_candidates_per_year: int = 100,
) -> pd.DataFrame:
    """
    For each year in [start_year, end_year], scan up to max_candidates_per_year
    Kaggle songs (sorted by rank), and select up to max_songs_per_year songs
    that have a matching Billboard track with usable chord/tonic data.

    Returns a DataFrame with at least: year, decade, title, artist, rank.
    """
    rows = []

    for year in range(start_year, end_year + 1):
        logger.info(
            "Selecting up to %d usable songs for year %d.",
            max_songs_per_year,
            year,
        )

        df_y = df[df["year"] == year].copy()
        if df_y.empty:
            logger.warning("No Kaggle songs for year %d.", year)
            continue

        df_y_sorted = df_y.sort_values("rank").head(max_candidates_per_year)
        n_candidates = len(df_y_sorted)

        usable_rows: List[pd.Series] = []

        for _, row in df_y_sorted.iterrows():
            track_id = match_kaggle_row_to_billboard_track(row, billboard_lookup)
            if track_id is None or track_id not in tracks:
                continue

            track = tracks[track_id]
            chords = get_chords_for_billboard_track(track)
            if len(chords) < 2:
                continue

            row_copy = row.copy()
            row_copy["decade"] = (int(year) // 10) * 10
            usable_rows.append(row_copy)

            if len(usable_rows) >= max_songs_per_year:
                break

        if usable_rows:
            df_year = pd.DataFrame(usable_rows)
            rows.append(df_year)
            logger.info(
                "Selected %d usable songs out of %d candidates for year %d.",
                len(df_year),
                n_candidates,
                year,
            )
        else:
            logger.warning(
                "No usable songs (with chords & tonic) found for year %d.",
                year,
            )

    if not rows:
        raise ValueError("No usable songs found in the requested year range.")

    out = pd.concat(rows, ignore_index=True)
    logger.info(
        "Total selected songs across years %d-%d: %d",
        start_year,
        end_year,
        len(out),
    )
    return out


# ---------------------------------------------------------------------------
# 6. Song-level result object & analysis pipeline
# ---------------------------------------------------------------------------

@dataclass
class SongChainResult:
    title: str
    artist: str
    year: int
    decade: int
    n_states: int
    spectral_gap: float
    lambda2_abs: float
    power_iters: int
    tol: float
    t_spec: float
    converged: bool


def analyze_song_row(
    row: pd.Series,
    tracks: Dict[str, "mirdata.datasets.billboard.Track"],
    billboard_lookup: Dict[Tuple[str, str], List[str]],
    power_tol: float = 1e-8,
    alpha: float = 1e-3,
    checkpoint_dir: Optional[Path] = None,
    debug_example: bool = False,
    min_states: int = 3,   # <-- NEW: require more than 2 states
) -> Optional[SongChainResult]:
    """
    Full pipeline for one Kaggle song.

    Songs with n_states < min_states are skipped. In particular,
    with min_states=3 we exclude 1- and 2-state chains, which are
    the main source of pathological tiny spectral gaps.

    When debug_example=True, print the chord sequence and transition
    matrix to give an interpretable example of what the Markov chain
    looks like for this song.
    """
    title = str(row["title"])
    artist = str(row["artist"])
    year = int(row["year"])
    decade = int(row["decade"])

    song_key = make_song_key(year, artist, title)
    logger.info("Analyzing song %s - %s (%d, decade %d).", artist, title, year, decade)

    track_id = match_kaggle_row_to_billboard_track(row, billboard_lookup)
    if track_id is None or track_id not in tracks:
        logger.warning(
            "No McGill match for %d - %s - %s. Skipping.",
            year,
            artist,
            title,
        )
        return None

    track = tracks[track_id]
    chords = get_chords_for_billboard_track(track)
    if len(chords) < 2:
        logger.warning(
            "Too few chords from McGill for %d - %s - %s. Skipping.",
            year,
            artist,
            title,
        )
        return None

    try:
        P, states = build_transition_matrix_from_chords(chords, alpha=alpha)
    except Exception as exc:
        logger.exception(
            "Error building transition matrix for %s - %s (%d): %s",
            artist,
            title,
            year,
            exc,
        )
        return None

    n_states = len(states)
    if n_states < min_states:
        logger.info(
            "Skipping song %s - %s (%d): only %d states (< %d).",
            artist,
            title,
            year,
            n_states,
            min_states,
        )
        return None

    # Optional interpretability: print chords + transition matrix for a couple songs
    if debug_example:
        print("\n=== Example Markov chain for song ===")
        print(f"{artist} - {title} ({year}, decade {decade})")
        print("\nSimplified roman-numeral chord sequence (first ~32 chords):")
        print(" ".join(chords[:32]))
        print(f"\nNumber of unique chord states: {n_states}")
        P_df = pd.DataFrame(P, index=states, columns=states)
        if n_states <= 12:
            print("\nTransition matrix P (rows: FROM, cols: TO):")
            print(P_df.round(3))
        else:
            print("\nTransition matrix P (top-left 12x12 block):")
            print(P_df.iloc[:12, :12].round(3))
        pi_preview = compute_stationary_distribution_linear(P)
        pi_df = pd.DataFrame({"state": states, "pi": pi_preview})
        pi_df = pi_df.sort_values("pi", ascending=False)
        print("\nTop 10 stationary chord probabilities:")
        print(pi_df.head(10).to_string(index=False))

    # Save per-song checkpoints: chords and transition matrix
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chords_path = checkpoint_dir / f"chords_{song_key}.txt"
        P_path = checkpoint_dir / f"transition_{song_key}.npz"
        try:
            with chords_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(chords))
            np.savez(P_path, P=P, states=np.array(states, dtype=object))
            logger.debug(
                "Saved chord sequence and transition matrix for %s to %s and %s.",
                song_key,
                chords_path,
                P_path,
            )
        except Exception as exc:
            logger.exception(
                "Failed to save checkpoints for %s: %s", song_key, exc
            )

    pi_exact = compute_stationary_distribution_linear(P)
    _, power_iters, converged = power_method_to_stationary(
        P, pi_exact=pi_exact, tol=power_tol
    )

    lambda2_abs, gap = compute_spectral_gap(P)
    t_spec = spectral_mixing_time_estimate(gap, tol=power_tol, a2_estimate=1.0)

    logger.info(
        "Song analyzed: n_states=%d, gap=%.4f, |lambda2|=%.4f, iters=%d, t_spec=%.1f, converged=%s",
        n_states,
        gap,
        lambda2_abs,
        power_iters,
        t_spec,
        converged,
    )

    return SongChainResult(
        title=title,
        artist=artist,
        year=year,
        decade=decade,
        n_states=n_states,
        spectral_gap=gap,
        lambda2_abs=lambda2_abs,
        power_iters=power_iters,
        tol=power_tol,
        t_spec=t_spec,
        converged=converged,
    )


def run_song_level_analysis(
    df_sel: pd.DataFrame,
    tracks: Dict[str, "mirdata.datasets.billboard.Track"],
    billboard_lookup: Dict[Tuple[str, str], List[str]],
    power_tol: float = 1e-8,
    alpha: float = 1e-3,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = True,
    n_example_transition_matrices: int = 2,
    min_states: int = 3,   # <-- pass through state threshold
) -> pd.DataFrame:
    """
    Driver for song-level analysis on a pre-selected set of songs (df_sel).

    - Optionally resumes from checkpoint_dir/song_results.csv if resume=True.
    - After each successfully analyzed song, writes updated song_results.csv.
    - For interpretability, prints up to n_example_transition_matrices
      example transition matrices with chord labels.
    """
    logger.info("Starting song-level analysis for %d selected songs.", len(df_sel))

    if df_sel.empty:
        logger.error("df_sel is empty; no songs to analyze.")
        return pd.DataFrame()

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    song_results_path: Optional[Path] = None
    df_results_existing: Optional[pd.DataFrame] = None
    processed_keys: set = set()

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        song_results_path = checkpoint_dir / "song_results.csv"
        if resume and song_results_path.exists():
            df_results_existing = pd.read_csv(song_results_path)
            logger.info(
                "Loaded %d existing song-level results from %s.",
                len(df_results_existing),
                song_results_path,
            )
            if not df_results_existing.empty:
                df_results_existing["year"] = pd.to_numeric(
                    df_results_existing["year"], errors="coerce"
                ).astype("Int64")
                processed_keys = {
                    (int(row["year"]), str(row["artist"]), str(row["title"]))
                    for _, row in df_results_existing.iterrows()
                    if pd.notnull(row["year"])
                }
        else:
            logger.info("No existing song-level checkpoint found or resume=False.")

    results_new: List[Dict] = []
    examples_remaining = n_example_transition_matrices

    for _, row in df_sel.iterrows():
        key = (int(row["year"]), str(row["artist"]), str(row["title"]))
        if key in processed_keys:
            logger.info(
                "Skipping already processed song %s - %s (%d).",
                key[1],
                key[2],
                key[0],
            )
            continue

        debug_example = examples_remaining > 0
        res = analyze_song_row(
            row,
            tracks=tracks,
            billboard_lookup=billboard_lookup,
            power_tol=power_tol,
            alpha=alpha,
            checkpoint_dir=checkpoint_dir,
            debug_example=debug_example,
            min_states=min_states,
        )
        if res is not None:
            results_new.append(asdict(res))
            if debug_example:
                examples_remaining -= 1

            if song_results_path is not None:
                try:
                    df_new = pd.DataFrame(results_new)
                    if df_results_existing is not None and not df_results_existing.empty:
                        df_save = pd.concat(
                            [df_results_existing, df_new], ignore_index=True
                        )
                    else:
                        df_save = df_new
                    df_save.to_csv(song_results_path, index=False)
                    logger.debug(
                        "Wrote updated song_results.csv with %d rows to %s.",
                        len(df_save),
                        song_results_path,
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to write song_results checkpoint: %s", exc
                    )

    df_new_final = pd.DataFrame(results_new)
    if df_results_existing is not None and not df_results_existing.empty:
        df_results = pd.concat(
            [df_results_existing, df_new_final], ignore_index=True
        ).reset_index(drop=True)
    else:
        df_results = df_new_final.reset_index(drop=True)

    logger.info("Song-level analysis produced %d total results.", len(df_results))
    return df_results


# ---------------------------------------------------------------------------
# 7. Decade-level aggregation from chords (aggregated Markov chains)
# ---------------------------------------------------------------------------

def aggregate_stationary_by_decade_from_tracks(
    df_sel: pd.DataFrame,
    tracks: Dict[str, "mirdata.datasets.billboard.Track"],
    billboard_lookup: Dict[Tuple[str, str], List[str]],
    alpha: float = 1e-3,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Build aggregated Markov chains per decade by summing transition counts
    over all matched McGill tracks in that decade.

    Returns
    -------
    df_decade : DataFrame with columns:
        decade, n_states, spectral_gap, lambda2_abs, n_songs.
    pi_by_decade : dict
        decade -> { chord_label: stationary_prob }.
    """
    logger.info("Starting decade-level aggregated Markov chain construction.")

    counts_by_decade: Dict[int, Dict[Tuple[str, str], float]] = {}
    state_sets_by_decade: Dict[int, set] = {}
    n_songs_by_decade: Dict[int, int] = {}

    for _, row in df_sel.iterrows():
        decade = int(row["decade"])
        year = int(row["year"])
        artist = str(row["artist"])
        title = str(row["title"])

        track_id = match_kaggle_row_to_billboard_track(row, billboard_lookup)
        if track_id is None or track_id not in tracks:
            continue

        track = tracks[track_id]
        chords = get_chords_for_billboard_track(track)
        if len(chords) < 2:
            continue

        state_sets_by_decade.setdefault(decade, set()).update(chords)
        n_songs_by_decade[decade] = n_songs_by_decade.get(decade, 0) + 1

        counts_dict = counts_by_decade.setdefault(decade, {})
        for c_from, c_to in zip(chords[:-1], chords[1:]):
            key = (c_from, c_to)
            counts_dict[key] = counts_dict.get(key, 0.0) + 1.0

    decade_stats = []
    pi_by_decade: Dict[int, Dict[str, float]] = {}

    for decade, state_set in state_sets_by_decade.items():
        states = sorted(state_set)
        idx = {ch: i for i, ch in enumerate(states)}
        n = len(states)

        C = np.zeros((n, n), dtype=float)
        counts_dict = counts_by_decade[decade]
        for (c_from, c_to), val in counts_dict.items():
            i = idx[c_from]
            j = idx[c_to]
            C[i, j] += val

        P = C + alpha
        P = P / P.sum(axis=1, keepdims=True)

        pi = compute_stationary_distribution_linear(P)
        lambda2_abs, gap = compute_spectral_gap(P)

        decade_stats.append(
            {
                "decade": decade,
                "n_states": n,
                "spectral_gap": gap,
                "lambda2_abs": lambda2_abs,
                "n_songs": n_songs_by_decade.get(decade, 0),
            }
        )
        pi_by_decade[decade] = {states[i]: float(pi[i]) for i in range(n)}

        logger.info(
            "Decade %d: n_states=%d, n_songs=%d, gap=%.4f, |lambda2|=%.4f",
            decade,
            n,
            n_songs_by_decade.get(decade, 0),
            gap,
            lambda2_abs,
        )

    if not decade_stats:
        logger.warning(
            "No decade-level transition counts were accumulated; "
            "returning empty decade stats."
        )
        return pd.DataFrame(), {}

    df_decade = pd.DataFrame(decade_stats).sort_values("decade")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        decade_chain_path = checkpoint_dir / "decade_chain_stats.csv"
        pi_by_decade_path = checkpoint_dir / "pi_by_decade.json"
        try:
            df_decade.to_csv(decade_chain_path, index=False)
            with pi_by_decade_path.open("w", encoding="utf-8") as f:
                json.dump(pi_by_decade, f, indent=2)
            logger.info(
                "Saved decade-level chain stats to %s and pi_by_decade to %s.",
                decade_chain_path,
                pi_by_decade_path,
            )
        except Exception as exc:
            logger.exception(
                "Failed to save decade-level checkpoints: %s", exc
            )

    return df_decade, pi_by_decade


# ---------------------------------------------------------------------------
# 8. Regression & plotting scaffolding
# ---------------------------------------------------------------------------

def regression_tspec_vs_iters(
    df_results: pd.DataFrame,
    filter_converged: bool = True,
    max_tspec: Optional[float] = None,
) -> Dict[str, float]:
    """
    Simple linear regression: t_PM ~ a * t_spec + b.

    Optionally:
      - filter to converged songs only,
      - trim out extremely large t_spec values (which are usually
        tiny-gap edge cases).
    """
    if df_results.empty:
        logger.warning("df_results is empty; regression not possible.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}

    df = df_results.copy()

    if filter_converged and "converged" in df.columns:
        df = df[df["converged"]]

    if max_tspec is not None:
        df = df[df["t_spec"] <= max_tspec]

    if df.empty:
        logger.warning("No rows left after regression filtering.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}

    x = df["t_spec"].values
    y = df["power_iters"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        logger.warning("Not enough finite points for regression.")
        return {"slope": np.nan, "intercept": np.nan, "R2": np.nan}

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    logger.info(
        "Regression (filtered=%s, max_tspec=%s): t_PM ~ %.4f * t_spec + %.4f (R^2=%.3f).",
        filter_converged,
        max_tspec,
        slope,
        intercept,
        r2,
    )
    return {"slope": slope, "intercept": intercept, "R2": r2}


def plot_tspec_vs_iters_loglog(df_results: pd.DataFrame) -> None:
    """
    Scatter plot (log–log) of t_spec vs t_PM, color-coded by number of states,
    and with marker shapes indicating convergence (circle = converged, x = not).
    """

    if df_results.empty:
        logger.warning("df_results is empty; skipping t_spec vs iters plot.")
        return

    # Filter positive values for log scale
    df = df_results[
        (df_results["t_spec"] > 0) &
        (df_results["power_iters"] > 0) &
        (df_results["n_states"] > 0)
    ].copy()

    if df.empty:
        logger.warning("No positive t_spec / power_iters values to plot.")
        return

    # Prepare data
    x = np.log10(df["t_spec"])
    y = np.log10(df["power_iters"])
    states = df["n_states"]
    converged = df["converged"]

    plt.figure(figsize=(7, 5))

    # Converged points: circles
    mask_conv = converged == True
    sc1 = plt.scatter(
        x[mask_conv], y[mask_conv],
        c=states[mask_conv],
        cmap="viridis",
        alpha=0.75,
        s=45,
        label="Converged",
        edgecolors="k"
    )

    # Non-converged points: X markers
    mask_non = ~mask_conv
    if mask_non.sum() > 0:
        sc2 = plt.scatter(
            x[mask_non], y[mask_non],
            c=states[mask_non],
            cmap="viridis",
            alpha=0.75,
            s=70,
            marker="x",
            label="Hit max iterations"
        )

    # Colorbar shows number of states
    cbar = plt.colorbar(sc1)
    cbar.set_label("Number of chord states (n_states)", fontsize=10)

    plt.xlabel(r"$\log_{10}(t_{\mathrm{spec}})$", fontsize=12)
    plt.ylabel(r"$\log_{10}(t_{\mathrm{PM}})$", fontsize=12)
    plt.title("Spectral Mixing-Time Estimate vs Power-Method Iterations\nColor = number of chord states", fontsize=13)

    plt.legend()
    plt.tight_layout()
    plt.savefig("spectral_mixing_vs_power_iterations.png", dpi=150)
    plt.close()



def aggregate_by_decade_from_song_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Simple aggregation of mean spectral gap & mean iterations by decade.
    """
    if df_results.empty:
        logger.warning("df_results is empty; cannot aggregate by decade.")
        return pd.DataFrame()

    if "decade" not in df_results.columns:
        logger.error("'decade' column missing in df_results; cannot aggregate.")
        return pd.DataFrame()

    grp = df_results.groupby("decade")
    df_decade = grp.agg(
        mean_gap=("spectral_gap", "mean"),
        std_gap=("spectral_gap", "std"),
        mean_iters=("power_iters", "mean"),
        std_iters=("power_iters", "std"),
        n_songs=("title", "count"),
    ).reset_index()
    logger.info("Aggregated decade-level stats from song results:\n%s", df_decade)
    return df_decade


def aggregate_by_year_from_song_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregation of mean spectral gap & mean iterations by year, with
    standard errors and approximate 95% CIs.
    """
    if df_results.empty:
        logger.warning("df_results is empty; cannot aggregate by year.")
        return pd.DataFrame()

    grp = df_results.groupby("year")
    df_year = grp.agg(
        mean_gap=("spectral_gap", "mean"),
        std_gap=("spectral_gap", "std"),
        mean_iters=("power_iters", "mean"),
        std_iters=("power_iters", "std"),
        n_songs=("title", "count"),
    ).reset_index()

    df_year["gap_se"] = df_year["std_gap"] / np.sqrt(df_year["n_songs"])
    df_year["iters_se"] = df_year["std_iters"] / np.sqrt(df_year["n_songs"])

    z = 1.96
    df_year["gap_ci_lo"] = df_year["mean_gap"] - z * df_year["gap_se"]
    df_year["gap_ci_hi"] = df_year["mean_gap"] + z * df_year["gap_se"]
    df_year["iters_ci_lo"] = df_year["mean_iters"] - z * df_year["iters_se"]
    df_year["iters_ci_hi"] = df_year["mean_iters"] + z * df_year["iters_se"]

    logger.info("Aggregated year-level stats from song results:\n%s", df_year)
    return df_year


def plot_decade_trends(df_decade: pd.DataFrame) -> None:
    """
    Plot decade-level trends of mean gap and mean power-method iterations.
    """
    if df_decade.empty:
        logger.warning("df_decade is empty; skipping decade trend plot.")
        return

    decades = df_decade["decade"].values
    mean_gap = df_decade["mean_gap"].values
    mean_iters = df_decade["mean_iters"].values

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xlabel("Decade start year")
    ax1.set_ylabel("Mean spectral gap", color="tab:blue")
    ax1.plot(decades, mean_gap, marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean power-method iterations", color="tab:red")
    ax2.plot(decades, mean_iters, marker="s", linestyle="--", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Decade-level trends in gap and mixing difficulty")
    fig.tight_layout()
    plt.savefig("decade_trends.png", dpi=150)
    plt.close()


def plot_year_trends(df_year: pd.DataFrame) -> None:
    """
    Plot year-level trends of mean gap and mean power-method iterations.
    """
    if df_year.empty:
        logger.warning("df_year is empty; skipping year trend plot.")
        return

    years = df_year["year"].values
    mean_gap = df_year["mean_gap"].values
    mean_iters = df_year["mean_iters"].values

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean spectral gap", color="tab:blue")
    ax1.plot(years, mean_gap, marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean power-method iterations", color="tab:red")
    ax2.plot(years, mean_iters, marker="s", linestyle="--", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Year-level trends in gap and mixing difficulty")
    fig.tight_layout()
    plt.savefig("year_trends.png", dpi=150)
    plt.close()


def plot_histograms(df_results: pd.DataFrame) -> None:
    """
    Plot histograms of spectral gaps and power-method iterations.
    """
    if df_results.empty:
        logger.warning("df_results is empty; skipping histograms.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(df_results["spectral_gap"].values, bins=30, alpha=0.8)
    plt.xlabel("Spectral gap")
    plt.ylabel("Count of songs")
    plt.title("Distribution of spectral gaps across songs")
    plt.tight_layout()
    plt.savefig("spectral_gap_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df_results["power_iters"].values, bins=30, alpha=0.8)
    plt.xlabel("Power-method iterations")
    plt.ylabel("Count of songs")
    plt.title("Distribution of power-method iterations")
    plt.tight_layout()
    plt.savefig("power_iterations_distribution.png", dpi=150)
    plt.close()


def plot_gap_vs_states(df_results: pd.DataFrame) -> None:
    """
    Scatter plot of spectral gap vs number of states to see if larger
    state spaces tend to have smaller gaps.
    """
    if df_results.empty:
        logger.warning("df_results is empty; skipping gap vs states plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(df_results["n_states"], df_results["spectral_gap"], alpha=0.7)
    plt.xlabel("Number of chord states in song")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap vs number of chord states")
    plt.tight_layout()
    plt.savefig("spectral_gap_vs_states.png", dpi=150)
    plt.close()


def print_small_gap_songs(df_results: pd.DataFrame, k: int = 10) -> None:
    """
    Print the top-k songs with the smallest spectral gaps for inspection.
    """
    if df_results.empty:
        return
    df_sorted = df_results.sort_values("spectral_gap").head(k)
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


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------

def main():
    # --- User configuration ---
    KAGGLE_CSV_PATH = "all_songs_data.csv"
    CHECKPOINT_DIR = Path("checkpoints")

    YEAR_START = 1959
    YEAR_END = 1991
    MAX_SONGS_PER_YEAR = 20      # try to pull more songs per year if possible
    MAX_CANDIDATES_PER_YEAR = 100

    POWER_TOL = 1e-6
    ALPHA = 1e-3
    MIN_STATES_FOR_ANALYSIS = 3  # <-- MORE THAN 2 STATES

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    selected_songs_path = CHECKPOINT_DIR / "selected_songs.csv"
    song_results_path_final = CHECKPOINT_DIR / "song_results_final.csv"
    decade_song_agg_path = CHECKPOINT_DIR / "decade_song_agg.csv"
    year_song_agg_path = CHECKPOINT_DIR / "year_song_agg.csv"

    # 1) Load Kaggle CSV
    df_kaggle = load_kaggle_top_songs(KAGGLE_CSV_PATH)

    # Restrict to relevant years early
    df_kaggle = df_kaggle[
        (df_kaggle["year"] >= YEAR_START) & (df_kaggle["year"] <= YEAR_END)
    ].copy()

    # 2) Initialize & load McGill Billboard dataset via mirdata
    dataset, tracks = initialize_billboard_dataset()
    billboard_lookup = build_billboard_lookup(tracks)

    # 3) Select songs per year with chord availability
    df_sel = select_songs_year_range(
        df=df_kaggle,
        tracks=tracks,
        billboard_lookup=billboard_lookup,
        start_year=YEAR_START,
        end_year=YEAR_END,
        max_songs_per_year=MAX_SONGS_PER_YEAR,
        max_candidates_per_year=MAX_CANDIDATES_PER_YEAR,
    )

    # Save selected songs checkpoint
    try:
        df_sel.to_csv(selected_songs_path, index=False)
        logger.info("Saved selected songs to %s.", selected_songs_path)
    except Exception as exc:
        logger.exception("Failed to save selected songs checkpoint: %s", exc)

    # 4) Song-level analysis
    df_results = run_song_level_analysis(
        df_sel=df_sel,
        tracks=tracks,
        billboard_lookup=billboard_lookup,
        power_tol=POWER_TOL,
        alpha=ALPHA,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=True,
        n_example_transition_matrices=5,
        min_states=MIN_STATES_FOR_ANALYSIS,
    )

    if df_results.empty:
        logger.error("No songs were successfully analyzed; skipping regression and plots.")
    else:
        # Save final song-level results
        try:
            df_results.to_csv(song_results_path_final, index=False)
            logger.info("Saved final song-level results to %s.", song_results_path_final)
        except Exception as exc:
            logger.exception("Failed to save final song-level results: %s", exc)

        print("\nSong-level results (head):")
        print(df_results.head())

        # Print some of the most problematic tiny-gap songs
        print_small_gap_songs(df_results, k=10)

        # 5) Regression: can t_spec predict t_PM (on the reasonable subset)?
        stats_all = regression_tspec_vs_iters(
            df_results, filter_converged=False, max_tspec=None
        )
        print("\nRegression on all songs (including non-converged and huge t_spec):")
        print(stats_all)

        stats_filtered = regression_tspec_vs_iters(
            df_results, filter_converged=True, max_tspec=1e4
        )
        print("\nRegression on converged songs with t_spec <= 1e4:")
        print(stats_filtered)

        # 6) Plots: scatter (including outliers), histograms, etc.
        plot_tspec_vs_iters_loglog(df_results)
        plot_histograms(df_results)
        plot_gap_vs_states(df_results)

        # 7) Year-level aggregation
        df_year = aggregate_by_year_from_song_results(df_results)
        if not df_year.empty:
            try:
                df_year.to_csv(year_song_agg_path, index=False)
                logger.info("Saved year-level averages to %s.", year_song_agg_path)
            except Exception as exc:
                logger.exception(
                    "Failed to save year-level averages from song results: %s", exc
                )

            print("\nYear-level averages (head):")
            print(df_year.head())
            print("\nYear-level averages (tail):")
            print(df_year.tail())

            plot_year_trends(df_year)

        # 8) Decade-level averages from song-level results
        df_decade_song_agg = aggregate_by_decade_from_song_results(df_results)
        if not df_decade_song_agg.empty:
            try:
                df_decade_song_agg.to_csv(decade_song_agg_path, index=False)
                logger.info(
                    "Saved decade-level averages from song results to %s.",
                    decade_song_agg_path,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to save decade-level averages from song results: %s",
                    exc,
                )

            print("\nDecade-level averages from song results:")
            print(df_decade_song_agg)

            plot_decade_trends(df_decade_song_agg)

    # 9) Aggregated Markov chains per decade (more principled averaging)
    df_decade_chain, pi_by_decade = aggregate_stationary_by_decade_from_tracks(
        df_sel=df_sel,
        tracks=tracks,
        billboard_lookup=billboard_lookup,
        alpha=ALPHA,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    if df_decade_chain.empty:
        logger.warning("Decade-level Markov-chain stats are empty.")
    else:
        print("\nDecade-level Markov-chain stats from aggregated counts:")
        print(df_decade_chain)


if __name__ == "__main__":
    main()
