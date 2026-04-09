# Speed of light in m/s
C: float = 299792458.0

# BeiDou frequency bands differ between RINEX 3.01/3.02 and 3.03+
# RINEX 3.01 / 3.02: L1=B1I, L6=B3I, L7=B2I
BEIDOU_FREQ_BANDS_LEGACY: dict[str, float] = {
    "L1": 1561.098e6,
    "L2": 1561.098e6,
    "L6": 1268.52e6,
    "L7": 1207.14e6,
}

# RINEX 3.03 / 3.04 / 3.05: L1=B1C, L2=B1I, L5=B2a, L6=B3I, L7=B2I, L8=B2ab
BEIDOU_FREQ_BANDS_MODERN: dict[str, float] = {
    "L1": 1575.42e6,
    "L2": 1561.098e6,
    "L5": 1176.45e6,
    "L6": 1268.52e6,
    "L7": 1207.14e6,
    "L8": 1191.795e6,
}

FREQ_BANDS: dict[str, dict] = {
    "G": {"L1": 1575.42e6, "L2": 1227.60e6, "L5": 1176.45e6},
    "R": {
        "L1": lambda n: (1602 + n * 0.5625) * 1e6,
        "L2": lambda n: (1246 + n * 0.4375) * 1e6,
    },
    "E": {"L1": 1575.42e6, "L5": 1176.45e6, "L7": 1207.14e6, "L8": 1191.795e6},
    "C": BEIDOU_FREQ_BANDS_MODERN,
}

# Priorities for frequency pairs
PHASE_FREQ_PRIORITY = {
    "G": [("L2", "L1"), ("L5", "L1")],
    "E": [("L5", "L1"), ("L7", "L1"), ("L8", "L1")],
    "C": [
        ("L7", "L1"),
        ("L7", "L2"),
        ("L6", "L1"),
        ("L6", "L2"),
        ("L5", "L1"),
        ("L5", "L2"),
    ],
    "R": [("L2", "L1")],
}

QUALITY_BAND_PAIR_PRIORITY = {
    "G": [("L5", "L1"), ("L2", "L1")],
    "E": [("L5", "L1"), ("L8", "L1"), ("L7", "L1")],
    "C": [
        ("L5", "L2"),
        ("L6", "L2"),
        ("L7", "L2"),
        ("L7", "L1"),
        ("L6", "L1"),
        ("L5", "L1"),
    ],
    "R": [("L2", "L1")],
}

BAND_PAIR_PRIORITY_BY_MODE = {
    "legacy": PHASE_FREQ_PRIORITY,
    "quality": QUALITY_BAND_PAIR_PRIORITY,
    "availability": {
        system: list(
            dict.fromkeys(
                QUALITY_BAND_PAIR_PRIORITY.get(system, [])
                + PHASE_FREQ_PRIORITY.get(system, [])
            )
        )
        for system in set(PHASE_FREQ_PRIORITY) | set(QUALITY_BAND_PAIR_PRIORITY)
    },
}
CODE_FREQ_PRIORITY = {
    "G": [("C2", "C1"), ("C5", "C1")],
    "E": [("C5", "C1"), ("C7", "C1"), ("C8", "C1")],
    "C": [
        ("C7", "C1"),
        ("C7", "C2"),
        ("C6", "C1"),
        ("C6", "C2"),
        ("C5", "C1"),
        ("C5", "C2"),
    ],
    "R": [("C2", "C1")],
}

# Priorities for channel suffixes
PHASE_CHAN_PRIORITY = ["C", "L", "S", "I", "Q", "W", "X", "P"]
CODE_CHAN_PRIORITY = ["C", "L", "S", "I", "Q", "W", "X", "P"]
