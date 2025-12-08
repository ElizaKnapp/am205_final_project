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



def parse_harte_chord_label(label: str):
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


def simplify_roman_sequence(roman_chords):
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
