import polars as pl
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
import numpy as np

# Costanti fisiche
C = 299792500.0  # Velocità della luce in m/s


@dataclass
class GNSSConstants:
    """Memorizza le costanti delle frequenze per diversi sistemi GNSS"""

    # Galileo
    GALILEO = {"L1": 1575.42e6, "L5": 1176.45e6, "L7": 1207.140e6, "L8": 1191.795e6}

    # BeiDou
    BEIDOU = {
        "0.01-0.02": {"L1": 1561.098e6, "L6": 1268.52e6, "L7": 1207.14e6},
        "0.03-0.05": {"L2": 1561.098e6, "L6": 1268.52e6, "L7": 1207.140e6},
    }

    # Altri sistemi
    GPS = {"L1": 1575.42e6, "L2": 1227.60e6}
    SBAS = {"L1": 1575.42e6, "L5": 1176.45e6}
    QZSS = {"L1": 1575.42e6, "L2": 1227.60e6}


def calculate_gflc(phase1: float, phase2: float, freq1: float, freq2: float) -> float:
    """Calcola la combinazione lineare geometry-free (GFLC)"""
    lambda1 = C / freq1
    lambda2 = C / freq2
    pr_to_tec = 1 / 40.308 * (freq1**2 * freq2**2) / (freq1**2 - freq2**2) / 1e16
    return (phase1 * lambda1 - phase2 * lambda2) * pr_to_tec


def get_satellite_id(system: str, prn: Union[int, str]) -> str:
    """Formatta l'ID del satellite con il prefisso del sistema"""
    prn_num = int(prn) if isinstance(prn, str) else prn
    return f"{system}{prn_num:02d}"


class GFLCCalculator:
    def __init__(
        self,
        obs: Dict[str, pl.DataFrame],
        obs_header: Dict,
        frequency_number: Optional[pl.DataFrame] = None,
        gnss_systems: List[str] = ["G", "R", "E", "C", "S", "J"],
    ):
        """
        Calcola GFLC per osservazioni GNSS usando Polars.

        Args:
            obs: Dizionario di Polars DataFrames con osservazioni per ogni sistema
            obs_header: Dizionario con metadati del file RINEX
            frequency_number: Polars DataFrame con numeri di frequenza per GLONASS
            gnss_systems: Lista di sistemi da processare (default: tutti)
        """
        self.obs = obs
        self.obs_header = obs_header
        self.frequency_number = frequency_number
        self.gnss_systems = gnss_systems
        self.constants = GNSSConstants()
        self.version = str(obs_header)

    def compute_gflc(self) -> Dict[str, pl.DataFrame]:
        """Calcola GFLC per tutti i sistemi richiesti"""
        results = {}

        if self.version.startswith("2"):
            results.update(self._process_rinex_v2())
        elif self.version.startswith("3"):
            results.update(self._process_rinex_v3())
        else:
            print(f"RINEX v{self.version} non supportato!")

        return results

    def _process_rinex_v2(self) -> Dict[str, pl.DataFrame]:
        """Processa dati RINEX v2.xx"""
        results = {}

        if "G" in self.gnss_systems and "GPS" in self.obs:
            results["GPS"] = self._process_gps_v2()

        if (
            "R" in self.gnss_systems
            and "GLONASS" in self.obs
            and self.frequency_number is not None
        ):
            results["GLONASS"] = self._process_glonass_v2()

        if "E" in self.gnss_systems and "Galileo" in self.obs:
            results["Galileo"] = self._process_galileo_v2()

        return results

    def _process_rinex_v3(self) -> Dict[str, pl.DataFrame]:
        """Processa dati RINEX v3.xx"""
        results = {}

        if "G" in self.gnss_systems and "GPS" in self.obs:
            results["GPS"] = self._process_gps_v3()

        if (
            "R" in self.gnss_systems
            and "GLONASS" in self.obs
            and self.frequency_number is not None
        ):
            results["GLONASS"] = self._process_glonass_v3()

        if "E" in self.gnss_systems and "Galileo" in self.obs:
            results["Galileo"] = self._process_galileo_v3()

        return results

    def _process_gps_v2(self) -> pl.DataFrame:
        """Processa dati GPS per RINEX v2"""
        df = self.obs["GPS"]
        required_cols = {"L1", "L2", "SatelliteID"}
        if required_cols.issubset(df.columns):
            return df.with_columns(
                [
                    pl.struct(["L1", "L2", "SatelliteID"])
                    .map(
                        lambda x: self._calculate_gps_gflc(
                            x["L1"],
                            x["L2"],
                            x["SatelliteID"],
                            self.constants.GPS["L1"],
                            self.constants.GPS["L2"],
                            "G",
                        )
                    )
                    .alias("gflc_prn")
                ]
            ).unnest("gflc_prn")
        return pl.DataFrame()

    def _process_glonass_v2(self) -> pl.DataFrame:
        """Processa dati GLONASS per RINEX v2"""
        df = self.obs["GLONASS"]
        required_cols = {"L1", "L2", "SatelliteID"}
        if required_cols.issubset(df.columns):
            return df.with_columns(
                [
                    pl.struct(["L1", "L2", "SatelliteID"])
                    .map(
                        lambda x: self._calculate_glonass_gflc(
                            x["L1"],
                            x["L2"],
                            x["SatelliteID"],
                            self.frequency_number,
                            "R",
                        )
                    )
                    .alias("gflc_prn")
                ]
            ).unnest("gflc_prn")
        return pl.DataFrame()

    @staticmethod
    def _calculate_gps_gflc(
        phase1: float,
        phase2: float,
        prn: Union[int, str],
        freq1: float,
        freq2: float,
        system: str,
    ) -> Dict[str, Union[float, str]]:
        """Calcola GFLC per GPS"""
        gflc = calculate_gflc(phase1, phase2, freq1, freq2)
        sat_id = get_satellite_id(system, prn)
        return {"gflc": gflc, "prn": sat_id}

    @staticmethod
    def _calculate_glonass_gflc(
        phase1: float,
        phase2: float,
        prn: Union[int, str],
        freq_numbers: pl.DataFrame,
        system: str,
    ) -> Dict[str, Union[float, str]]:
        """Calcola GFLC per GLONASS"""
        prn_num = int(prn) if isinstance(prn, str) else prn
        freq_row = freq_numbers.filter(pl.col("prn") == prn_num)

        if freq_row.is_empty():
            return {"gflc": np.nan, "prn": get_satellite_id(system, prn_num)}

        freq_num = freq_row["freqn"][0]
        freq1 = 1602 + freq_num * 0.5625
        freq2 = 1246 + freq_num * 0.4375

        gflc = calculate_gflc(phase1, phase2, freq1 * 1e6, freq2 * 1e6)
        sat_id = get_satellite_id(system, prn_num)
        return {"gflc": gflc, "prn": sat_id}
