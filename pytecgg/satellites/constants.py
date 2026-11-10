from datetime import date, datetime, timedelta, timezone
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
        "source",
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

# Leap seconds (UTC -> GPST) and the UTC date from which each value applies
# (0 before 1981-07-01). Used only when neither GNSSContext.leap_seconds nor the
# nav header LEAP SECONDS field gives a value.
LEAP_SECONDS_UTC_GPST: tuple[tuple[date, int], ...] = (
    (date(1981, 7, 1), 1),
    (date(1982, 7, 1), 2),
    (date(1983, 7, 1), 3),
    (date(1985, 7, 1), 4),
    (date(1988, 1, 1), 5),
    (date(1990, 1, 1), 6),
    (date(1991, 1, 1), 7),
    (date(1992, 7, 1), 8),
    (date(1993, 7, 1), 9),
    (date(1994, 7, 1), 10),
    (date(1996, 1, 1), 11),
    (date(1997, 7, 1), 12),
    (date(1999, 1, 1), 13),
    (date(2006, 1, 1), 14),
    (date(2009, 1, 1), 15),
    (date(2012, 7, 1), 16),
    (date(2015, 7, 1), 17),
    (date(2017, 1, 1), 18),
)

# The table is known correct before this date: IERS Bulletin C 72 (2026-07-06)
# announced no leap second at the end of December 2026; the next possible one is
# at the end of June 2027. Move it forward after each Bulletin C (January, July).
LEAP_SECONDS_VALID_UNTIL: date = date(2027, 7, 1)

# Leap-second count after the last table entry
DEFAULT_LEAP_SECONDS_UTC_GPST: int = LEAP_SECONDS_UTC_GPST[-1][1]

# BeiDou time (BDT) trails GPST by a fixed 14 s (BDS ICD, BDT epoch 2006-01-01).
# This is a constant defined by the ICD and never changes.
BDT_OFFSET_FROM_GPST: int = 14
