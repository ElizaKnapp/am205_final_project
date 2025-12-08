"""
Billboard dataset loader & lookup
"""
import mirdata
from collections import defaultdict
from utils import clean_string_for_match, parse_tonic_string

class BillboardDataset:
    def __init__(self):
        self.dataset = None
        self.tracks = None

    def _load_dataset(self):
        print("Initializing McGill Billboard dataset via mirdata.")
        self.dataset = mirdata.initialize("billboard")
        self.dataset.download()
        self.tracks = self.dataset.load_tracks()

    def _build_lookup(self):
        print("Building Billboard lookup by (title, artist).")
        self.lookup = defaultdict(list)
        for track_id, track in self.tracks.items():
            title_norm = clean_string_for_match(getattr(track, "title", "") or "")
            artist_norm = clean_string_for_match(getattr(track, "artist", "") or "")
            key = (title_norm, artist_norm)
            self.lookup[key].append(track_id)
        print(f"Billboard lookup built with {len(self.lookup)} unique (title, artist) keys.")

    def build(self):
        self._load_dataset()
        self._build_lookup()

    def match_song(self, title: str, artist: str):
        """
        Attempt to match a song to a billboard track_id using normalized
        (title, artist). If multiple matches, just return the first.
        If no match, return None.
        """
        clean_title = clean_string_for_match(title)
        clean_artist = clean_string_for_match(artist)
        # Recreate key
        key = (clean_title, clean_artist)
        track_ids = self.lookup.get(key, None)
        if track_ids is None:
            return track_ids
        return track_ids[0]
    
    def get_chords_for_track(self, track_id: str):
        """
        Get the chords for a track.
        """
        # Get chords
        track = self.tracks.get(track_id, None)
        if track is None:
            return None
        chord_data = getattr(track, "chords_majmin", None)
        if chord_data is None or getattr(chord_data, "labels", None) is None:
            print(f"Track {track_id} has no chords_majmin labels.")
            return []
        labels = chord_data.labels

        # Get tonics
        salami_metadata = getattr(track, "salami_metadata", None)
        if salami_metadata is None or "tonic" not in salami_metadata:
            print(f"Track {track_id} has no tonic metadata.")
            return []
        tonic = salami_metadata["tonic"]

        # Parse tonic
        tonic_pc, mode = parse_tonic_string(tonic)
        if tonic_pc is None:
            print(f"Track {track_id} has no valid tonic.")
            return []
        return labels, tonic_pc, mode