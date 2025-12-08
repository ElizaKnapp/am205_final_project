#####################
# Utility Functions #
#####################
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

def clean_string_for_match(s: str):
    """
    Clean a string for matching: lowercase, remove non-alphanumerics.
    """
    s = str(s).lower()
    return "".join(ch for ch in s if ch.isalnum())

def parse_tonic_string(tonic: str):
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

