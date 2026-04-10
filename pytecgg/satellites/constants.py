from datetime import datetime, timedelta, timezone
from typing import TypedDict, Optional
from dataclasses import dataclass


@dataclass
class ConstellationParams:
    time_system: str
    prefix: str
    time_offset: timedelta
    fields: list[str]


@dataclass
class GNSSConstants:
    """Physical and geodetic constants for GNSS constellations

    Attributes:
        gm (float): Geocentric gravitational constant [m^3*s^-2]
        we (float): Angular rotation rate of the Earth [rad*s^-1]
        a (float): Semi-major axis of Earth's ellipsoid [m]
        e (Optional[float]): Eccentricity of Earth's ellipsoid [dimensionless]
        f (Optional[float]): Flattening of Earth's ellipsoid [dimensionless]
        c20 (Optional[float]): Second degree zonal harmonic coefficient [dimensionless]
    """

    gm: float
    we: float
    a: float
    e: Optional[float] = None
    f: Optional[float] = None
    c20: Optional[float] = None


class EphemerisFields(TypedDict):
    GPS: list[str]
    BEIDOU: list[str]
    GLONASS: list[str]
    GALILEO: list[str]
    QZSS: list[str]
    SBAS: list[str]
    NAVIC: list[str]


EPHEMERIS_FIELDS: EphemerisFields = {
    "GPS": [
        "accuracy",
        "cic",
        "cis",
        "clock_bias",
        "clock_drift",
        "clock_drift_rate",
        "crc",
        "crs",
        "cuc",
        "cus",
        "deltaN",
        "e",
        "fitInt",
        "health",
        "i0",
        "idot",
        "iodc",
        "iode",
        "l2Codes",
        "l2p",
        "m0",
        "omega",
        "omega0",
        "omegaDot",
        "sqrta",
        "t_tm",
        "tgd",
        "toe",
    ],
    "BEIDOU": [
        "clock_bias",
        "clock_drift",
        "clock_drift_rate",
        "aode",
        "crs",
        "deltaN",
        "m0",
        "cuc",
        "e",
        "cus",
        "sqrta",
        "toe",
        "cic",
        "omega0",
        "cis",
        "i0",
        "crc",
        "omega",
        "omegaDot",
        "idot",
        "accuracy",
        "tgd1b1b3",
        "tgd2b2b3",
        "aodc",
    ],
    "GLONASS": [
        "accelX",
        "accelY",
        "accelZ",
        "channel",
        "clock_bias",
        "clock_drift",
        "clock_drift_rate",
        "health",
        "satPosX",
        "satPosY",
        "satPosZ",
        "velX",
        "velY",
        "velZ",
    ],
    "GALILEO": [
        "clock_bias",
        "clock_drift",
        "clock_drift_rate",
        "bgdE5aE1",
        "bgdE5bE1",
        "iodnav",
        "sisa",
        "health",
        "t_tm",
        "toe",
        "week",
        "sqrta",
        "e",
        "i0",
        "idot",
        "omega0",
        "omega",
        "omegaDot",
        "m0",
        "deltaN",
        "crc",
        "crs",
        "cuc",
        "cus",
        "cic",
        "cis",
    ],
    "QZSS": [
        "accuracy",
        "cic",
        "cis",
        "clock_bias",
        "clock_drift",
        "clock_drift_rate",
        "crc",
        "crs",
        "cuc",
        "cus",
        "deltaN",
        "e",
        "fitInt",
        "health",
        "i0",
        "idot",
        "iodc",
        "iode",
        "l2Codes",
        "l2p",
        "m0",
        "omega",
        "omega0",
        "omegaDot",
        "sqrta",
        "t_tm",
        "tgd",
        "toe",
        "week",
    ],
    "SBAS": [],
    "NAVIC": [],
}

CONSTELLATION_PARAMS = {
    "GPS": ConstellationParams(
        time_system="GPST",
        prefix="G",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["GPS"],
    ),
    "BEIDOU": ConstellationParams(
        time_system="BDT",
        prefix="C",
        time_offset=timedelta(hours=8),
        fields=EPHEMERIS_FIELDS["BEIDOU"],
    ),
    "GLONASS": ConstellationParams(
        time_system="UTC",
        prefix="R",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["GLONASS"],
    ),
    "GALILEO": ConstellationParams(
        time_system="GST",
        prefix="E",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["GALILEO"],
    ),
    "QZSS": ConstellationParams(
        time_system="QZSST",
        prefix="J",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["QZSS"],
    ),
    "NAVIC": ConstellationParams(
        time_system="IRNWT",
        prefix="I",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["NAVIC"],
    ),
    "SBAS": ConstellationParams(
        time_system="UTC",
        prefix="S",
        time_offset=timedelta(0),
        fields=EPHEMERIS_FIELDS["SBAS"],
    ),
}

GNSS_CONSTANTS: dict[str, GNSSConstants] = {
    "GPS": GNSSConstants(
        gm=3.986005e14,
        we=7.2921151467e-5,
        a=6378137,
        e=0.0818191908426215,
        f=1 / 298.257223563,
    ),
    "BEIDOU": GNSSConstants(
        gm=3.986004418e14, we=7.292115e-5, a=6378137, f=1 / 298.257222101
    ),
    "GLONASS": GNSSConstants(
        gm=3.9860044e14, we=7.292115e-5, a=6378136, c20=-1082.63e-6
    ),
    "GALILEO": GNSSConstants(gm=3.986004418e14, we=7.2921151467e-5, a=6378137),
    "QZSS": GNSSConstants(
        gm=3.986005e14,
        we=7.2921151467e-5,
        a=6378137,
    ),
    "NAVIC": GNSSConstants(
        gm=3.986005e14,
        we=7.2921151467e-5,
        a=6378137,
    ),
}

# Fields strictly required for Keplerian position computation.
# Used in prepare_ephemeris to validate ephemeris rows instead of the full
# EPHEMERIS_FIELDS lists, which also contain clock/bias fields that are
# optional for position calculation and may be legitimately absent.
KEPLERIAN_POSITION_FIELDS: list[str] = [
    "toe",
    "sqrta",
    "deltaN",
    "m0",
    "e",
    "omega",
    "cuc",
    "cus",
    "crc",
    "crs",
    "cic",
    "cis",
    "i0",
    "idot",
    "omega0",
    "omegaDot",
]

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

TOL_KEPLER: float = 0.001

# Earth radius in meters
RE: float = 6371000.0
