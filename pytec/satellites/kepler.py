import numpy as np


def kepler(e: float, mk: float, tol: float) -> float:
    """
    Computes the eccentric anomaly (ek) from the mean anomaly (mk)
    using Kepler's equation and a fixed-point iteration method.

    Parameters:
    - e: numerical eccentricity of the orbit (dimensionless)
    - mk: mean anomaly in radians
    - tol: convergence tolerance in arcseconds (")

    Returns:
    - ek: eccentric anomaly in radians
    """
    tol_rad = tol * (np.pi / 648000)

    e_prev = mk
    e_curr = mk + e * np.sin(e_prev)

    while abs(e_curr - e_prev) > tol_rad:
        e_prev, e_curr = e_curr, mk + e * np.sin(e_curr)

    return e_curr
