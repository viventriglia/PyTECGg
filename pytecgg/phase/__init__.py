# Earth radius in meters
RE: float = 6371000.0
# Speed of light in m/s
C: float = 299792458.0

FREQ_BANDS: dict[str, dict] = {
    "G": {"L1": 1575.42e6, "L2": 1227.60e6},
    "R": {
        "L1": lambda n: (1602 + n * 0.5625) * 1e6,
        "L2": lambda n: (1246 + n * 0.4375) * 1e6,
    },
    "E": {"L1": 1575.42e6, "L5": 1176.45e6},
    "C": {"L1": 1561.098e6, "L5": 1207.14e6},
}

OBS_MAPPING: dict[str, dict] = {
    "G": {"phase": {"L1": "L1C", "L2": "L2W"}},
    "R": {"phase": {"L1": "L1C", "L2": "L2C"}},
    "E": {"phase": {"L1": "L1X", "L5": "L5X"}},
    "C": {"phase": {"L1": "L1X", "L5": "L5X"}},
}
