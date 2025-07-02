from datetime import timedelta

# Constellation-specific parameters
CONSTELLATION_PARAMS = {
    "GPS": {"time_system": "GPST", "prefix": "G", "time_offset": timedelta(0)},
    "BeiDou": {
        "time_system": "BDT",
        "prefix": "C",
        "time_offset": timedelta(hours=8),
    },
    # TODO Add other constellations
}

EPHEMERIS_FIELDS: dict[str, list[str]] = {
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
    "BeiDou": [
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
    "GLONASS": [],
    "Galileo": [],
    "QZSS": [],
    "SBAS": [],
}

GNSS_CONSTANTS = {
    # gm: geocentric gravitational constant (m^3*s^-2)
    # we: angular rotation of Earth (rad*s^-1)
    # a: major-axis of Earth ellipsoid (m)
    # e: numeric eccentricity of ellipsoid (-)
    # f: flattening of Earth ellipsoid (-)
    # c20: second degree zonal harmonic coefficient (-)
    "GPS": {
        "gm": 3.986005e14,
        "we": 7.2921151467e-5,
        "a": 6378137,
        "e": 0.0818191908426215,
        "f": 1 / 298.257223563,
    },
    "GLONASS": {
        "gm": 3.9860044e14,
        "we": 7.292115e-5,
        "a": 6378136,
        "c20": -1082.63e-6,
    },
    "Galileo": {"gm": 3.986004418e14, "we": 7.2921151467e-5, "a": 6378137},
    "BeiDou": {
        "gm": 3.986004418e14,
        "we": 7.292115e-5,
        "a": 6378137,
        "f": 1 / 298.257222101,
    },
}

TOL_KEPLER = 0.001
