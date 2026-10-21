from dataclasses import dataclass, field
from typing import Any, Optional
import warnings

SUPPORTED_SYSTEMS = {
    "GPS": "G",
    "GLONASS": "R",
    "GALILEO": "E",
    "BEIDOU": "C",
    # "QZSS": "J",
    # "NAVIC": "I",
}


@dataclass
class GNSSContext:
    """
    Centralised configuration and state management for GNSS Total Electron Content (TEC) analysis.

    The GNSSContext acts as a single source of truth for the processing pipeline.
    It handles coordinate validation, constellation name normalisation, and
    maintains metadata required for geometric and frequency-dependent calculations.

    Attributes
    ----------
    receiver_pos : tuple[float, float, float]
        Receiver coordinates in the Earth-Centered, Earth-Fixed (ECEF) frame,
        expressed in meters (X, Y, Z).
    receiver_name : str
        Station identifier. Automatically normalised to lowercase and truncated
        to the first 4 characters (standard RINEX style).
    rinex_version : str
        The version of the source RINEX file (e.g., "2.11", "3.04").
    h_ipp : float
        The altitude of the thin-shell ionospheric model in meters.
        Default is 350,000 m (350 km).
    systems : list[str]
        List of active GNSS constellations, accepting full names or symbols:
        GPS (G), GLONASS (R), Galileo (E), BeiDou (C).
    glonass_channels : dict[str, int]
        Mapping of frequency channels for GLONASS satellites
        (e.g., {'R01': 1}). This attribute gets populated during ephemeris parsing.
    freq_meta : dict[str, Any]
        Additional metadata related to signal
        frequencies and observation types used during processing.
    leap_seconds : int, optional
        UTC -> GPST offset in seconds. Typically populated from the RINEX
        navigation header; falls back to ``DEFAULT_LEAP_SECONDS_UTC_GPST``
        from ``constants.py`` when left unset.

    Notes
    -----
    The class performs validation and sanitisation steps during `__post_init__`:

    * Validates that `receiver_pos` is a 3-element numeric structure.
    * Maps constellation names to standard symbols of supported constellations.
    * Issues a `UserWarning` if `h_ipp` falls outside the typical geophysical
        range of [250, 500] km.
    * Ensures at least one valid GNSS system is specified for processing.

    Examples
    --------
        >>> ctx = GNSSContext(
        ...     receiver_pos=(4444444.0, 1111111.0, 1234567.0),
        ...     receiver_name="GROT",
        ...     rinex_version="3.04",
        ...     systems=["GPS", "Galileo"],
        ... )
        >>> ctx.systems
        ['G', 'E']
        >>> ctx.receiver_name
        'grot'
    """

    receiver_pos: tuple[float, float, float]
    receiver_name: str
    rinex_version: str
    h_ipp: float = 350_000
    systems: list[str] = field(default_factory=list)
    glonass_channels: dict[str, int] = field(default_factory=dict)
    freq_meta: dict[str, Any] = field(default_factory=dict)
    # UTC -> GPST offset in seconds. Populate from the RINEX nav header via
    # parsing.read_rinex_nav_header(). Falls back to DEFAULT_LEAP_SECONDS_UTC_GPST
    # in constants.py if left unset.
    leap_seconds: Optional[int] = None

    def __post_init__(self):
        if (
            not isinstance(self.receiver_pos, (tuple, list))
            or len(self.receiver_pos) != 3
        ):
            raise ValueError(f"'receiver_pos' must be a tuple/list of three floats")

        normalized = []
        for s_ in self.systems:
            s_up = s_.upper()
            symbol_ = SUPPORTED_SYSTEMS.get(s_up, s_up)
            if symbol_ not in SUPPORTED_SYSTEMS.values():
                raise ValueError(
                    f"GNSS system '{s_}' not recognised, must be one of {list(SUPPORTED_SYSTEMS.keys()) + list(SUPPORTED_SYSTEMS.values())}"
                )
            normalized.append(symbol_)

        self.systems = list(set(normalized))
        if not self.systems:
            raise ValueError("At least one GNSS system must be specified in 'systems'")

        self.receiver_name = self.receiver_name.lower().strip()[:4]
        self.rinex_version = str(self.rinex_version).strip()

        if not (250_000 <= self.h_ipp <= 500_000):
            warnings.warn(
                f"'h_ipp' value of {self.h_ipp}m looks unusual, as typical values are between 200km and 500km. This is fine if intentional.",
                UserWarning,
            )

    @property
    def symbol_to_name(self) -> dict[str, str]:
        """Inverse mapping of GNSS systems active in the context."""
        full_mapping = {v: k for k, v in SUPPORTED_SYSTEMS.items()}
        return {s: full_mapping[s] for s in self.systems if s in full_mapping}
