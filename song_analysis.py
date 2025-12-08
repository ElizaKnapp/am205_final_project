"""
Class that performs song-level analysis.
"""

import numpy as np
import pandas as pd
from billboard import BillboardDataset

class SongAnalysis:
    def __init__(
        self, 
        song_data_path: str,
        billboard_dataset: BillboardDataset,
        year_start: int,    
        year_end: int,
        max_songs_per_year: int,
        max_candidates_per_year: int,
        power_method_tol: float,
        alpha: float,
        min_states: int,
    ):
        df = pd.read_csv(song_data_path)
        # Rename columns
        df = df.rename(
            columns={
                "Year": "year",
                "Rank": "rank",
                "Song Title": "title",
                "Artist": "artist",
            }
        )
        
        # Convert types
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        
        # Drop rows with missing critical info
        before = len(df)
        df = df.dropna(subset=["year", "rank", "title", "artist"])
        after = len(df)
        if after < before:
            print(f"Dropped {before - after} rows from Kaggle dataset due to missing year/rank/title/artist.")
        print(f"Loaded Kaggle dataset with {len(df)} rows after cleaning.")
        
        # Restrict to relevant years early
        df = df[
            (df["year"] >= year_start) & (df["year"] <= year_end)
        ].copy()

        # Set all the class attributes
        self.df = df
        self.billboard_dataset = billboard_dataset
        self.year_start = year_start
        self.year_end = year_end
        self.max_songs_per_year = max_songs_per_year
        self.max_candidates_per_year = max_candidates_per_year
        self.power_method_tol = power_method_tol
        self.alpha = alpha
        self.min_states = min_states


    def select_songs(self):
        """
        For each year in the range specified, scan up to max_candidates_per_year
        songs (sorted by rank), and from there, select up to max_songs_per_year songs
        that have a matching Billboard track with usable chord/tonic data.

        Returns a DataFrame with at least: year, decade, title, artist, rank.
        """        
        all_song_data = []
        # For each year, find enough candidates + actual songs to analyze
        for year in range(self.year_start, self.year_end + 1):
            print(f"Selecting up to {self.max_songs_per_year} usable songs for year {year}.")

            df_year = self.df[self.df["year"] == year].copy()
            if df_year.empty:
                print(f"No Kaggle songs for year {year}.")
                continue

            df_year_sorted = df_year.sort_values("rank").head(self.max_candidates_per_year)
            n_candidates = len(df_year_sorted)

            usable_rows = []
        
            for _, row in df_year_sorted.iterrows():
                track_id = self.billboard_dataset.match_song(row["title"], row["artist"])
                if track_id is None or track_id not in self.billboard_dataset.tracks:
                    continue

                track = self.billboard_dataset.tracks[track_id]
                chords = self.billboard_dataset.get_chords_for_track(track)
                if len(chords) < 2:
                    continue

                row_copy = row.copy()
                row_copy["decade"] = (int(year) // 10) * 10
                usable_rows.append(row_copy)
                # If we have found enough songs, stop looking
                if len(usable_rows) >= self.max_songs_per_year:
                    break
            # If we have enough songs, add this to the master data for the year
            if usable_rows:
                df_year = pd.DataFrame(usable_rows)
                all_song_data.append(df_year)
                print(f"Selected {len(df_year)} usable songs out of {n_candidates} candidates for year {year}.")
            else:
                print(f"No usable songs (with chords & tonic) found for year {year}.")

        if not all_song_data:
            raise ValueError("No usable songs found in the requested year range.")

        selected_songs = pd.concat(all_song_data, ignore_index=True)
        print(f"Total selected songs across years {self.year_start}-{self.year_end}: {len(final_df)}")
        return selected_songs
    
    def _build_transition_matrix_from_chords(
        self,
        chords: list,
    ):
        """
        Build row-stochastic transition matrix P (with smoothing) from a
        key-normalized chord sequence.

        States are unique chords in order of first appearance.        
        """
        if len(chords) < 2:
            raise ValueError("Chord sequence must have at least 2 chords to define transitions.")

        # First we build the unique chords and their indices in the list
        unique_chords = []
        index_of_chords = {}
        for chord in chords:
            if chord not in index_of_chords:
                index_of_chords[chord] = len(unique_chords)
                unique_chords.append(chord)

        # Now we find how much each chord transitions to each other chord
        n = len(unique_chords)
        counts = np.zeros((n, n), dtype=float)

        for c_from, c_to in zip(chords[:-1], chords[1:]):
            i = index_of_chords[c_from]
            j = index_of_chords[c_to]
            counts[i, j] += 1.0

        # Now we add the smoothing factor and normalize the rows
        P = counts + self.alpha
        P = P / P.sum(axis=1, keepdims=True)
        return P, unique_chords

    def _compute_stationary_distribution_linear(
        self,
        P: np.ndarray,
    ):
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

    def _power_method_to_stationary(
        self,
        P: np.ndarray,
        pi_exact: np.ndarray,
    ):
        """
        Run power method mu_{k+1}^T = mu_k^T P.
        """
        max_iter = 10**6
        mu = np.ones(P.shape[0])
        mu = mu / mu.sum()

        for k in range(1, max_iter + 1):
            mu = mu @ P
            dist = np.linalg.norm(mu - pi_exact, ord=1)
            if dist < self.power_method_tol:
                return mu, k, True

        print(f"Power method did not converge within {max_iter} iterations (final dist={dist:.3e}).")
        return mu, max_iter, False

    def _compute_spectral_gap(
        self,
        P: np.ndarray,
    ):
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

    def _spectral_mixing_time_estimate(
        self,
        spectral_gap: float,
    ):
        """
        Spectral mixing-time estimate t_spec ~ log(a2 / power_method_tol) / spectral_gap.
        """
        if spectral_gap <= 0:
            return float("inf")
        # We use a2 = 1.0 because it is a measure of the initial matrix to the stationary distribution which can have 1 as a baseline
        C = np.log(1.0 / self.power_method_tol)
        return float(C/spectral_gap)

    def _analyze_song_row(
        self,
        row: pd.Series,
    ):
        """
        Full pipeline for one song.
        """
        title = str(row["title"])
        artist = str(row["artist"])
        year = int(row["year"])
        decade = int(row["decade"])

        # Find the song based on the information
        track_id = self.billboard_dataset.match_song(title, artist)
        if track_id is None or track_id not in self.billboard_dataset.tracks:
            print(f"No McGill match for {year} - {artist} - {title}. Skipping.")
            return None

        # Get the chords for the song
        chords = self.billboard_dataset.get_chords_for_track(track_id)
        if len(chords) < 2:
            print(f"Too few chords from McGill for {year} - {artist} - {title}. Skipping.")
            return None

        try:
            P, states = self._build_transition_matrix_from_chords(chords)
        except Exception as e:
            print(f"Error building transition matrix for {artist} - {title} ({year}): {e}")
            return None

        n_states = len(states)
        if n_states < self.min_states:
            print(f"Skipping song {artist} - {title} ({year}): only {n_states} states (< {self.min_states}).")
            return None

        # Compute the stationary distribution exactly and then using the power method
        pi_exact = self._compute_stationary_distribution_linear(P)
        _, power_iters, converged = self._power_method_to_stationary(
            P, pi_exact=pi_exact,
        )

        lambda2_abs, spectral_gap = self._compute_spectral_gap(P)
        t_spec = self._spectral_mixing_time_estimate(spectral_gap)
        return {
            "title": title,
            "artist": artist,
            "year": year,
            "decade": decade,
            "n_states": n_states,
            "spectral_gap": spectral_gap,
            "lambda2_abs": lambda2_abs,
            "power_iters": power_iters,
            "tol": self.power_method_tol,
            "t_spec": t_spec,
            "converged": converged,
        }

    def analyze_songs(
        self, 
        selected_songs: pd.DataFrame, 
    ):
        """
        Main functionality for song-level analysis on a pre-selected set of songs (selected_songs).

        Returns a DataFrame with at least: year, decade, title, artist, rank.
        """
        print(f"Starting song-level analysis for {selected_songs} selected songs.")
        # End early if no results were able to be found
        if selected_songs.empty:
            print("df_sel is empty; no songs to analyze.")
            return pd.DataFrame()

        # Analyze each song
        results = []
        for _, row in selected_songs.iterrows():
            result_for_song = self._analyze_song_row(row)
            if result_for_song is not None:
                results.append(result_for_song)
        df_results = pd.DataFrame(results)
        return df_results