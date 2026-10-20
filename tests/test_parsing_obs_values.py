"""Parsed observation values must match the raw RINEX text, not just its shape."""

import datetime
import math

import polars as pl
import pytest

from pytecgg.parsing import read_rinex_obs

N_EPOCHS = 200


def _epoch(fields: list[str]) -> datetime.datetime:
    y, mo, d, h, mi = (int(v) for v in fields[:5])
    y += 2000 if y < 80 else (1900 if y < 100 else 0)
    return datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc) + datetime.timedelta(
        seconds=float(fields[5])
    )


def _values(sv: str, types: list[str], block: str, epoch: datetime.datetime):
    for k, obs in enumerate(types):
        text = block[16 * k : 16 * k + 14].strip()
        if text:
            yield epoch, sv, obs, float(text)


def _read_raw(path, max_epochs: int) -> pl.DataFrame:
    """Minimal independent RINEX 2/3 obs reader for the first `max_epochs` epochs."""
    lines = open(path, encoding="ascii", errors="replace").read().splitlines()
    version = float(lines[0][:9])
    types: dict[str, list[str]] = {}
    i, sys_ = 0, None
    while "END OF HEADER" not in lines[i]:
        label, data = lines[i][60:], lines[i][:60]
        if "# / TYPES OF OBSERV" in label:  # RINEX 2: one list for all systems
            types.setdefault("*", []).extend(data[6:].split())
        elif "SYS / # / OBS TYPES" in label:  # RINEX 3: one list per system
            sys_ = data[0] if data[0] != " " else sys_
            types.setdefault(sys_, []).extend(data[7:].split())
        i += 1
    i += 1

    rows, epochs = [], 0
    while i < len(lines) and epochs < max_epochs:
        head = lines[i]
        if version >= 3:
            if not head.startswith(">"):
                i += 1
                continue
            flag, nsat = int(head[31]), int(head[32:35])
            i += 1
            if flag > 1:
                i += nsat
                continue
            epoch = _epoch(head[2:29].split())
            for line in lines[i : i + nsat]:
                sv = f"{line[0]}{int(line[1:3]):02d}"
                rows += _values(sv, types[line[0]], line[3:].ljust(16 * len(types[line[0]])), epoch)
            i += nsat
        else:
            flag, nsat = int(head[28]), int(head[29:32])
            sats = head[32:68]
            while len(sats.rstrip()) < 3 * nsat:
                i += 1
                sats += lines[i][32:68]
            i += 1
            if flag > 1:
                i += nsat
                continue
            epoch = _epoch(head[1:26].split())
            n_lines = math.ceil(len(types["*"]) / 5)
            for s in range(nsat):
                sv_raw = sats[3 * s : 3 * s + 3]
                sv = f"{sv_raw[0].strip() or 'G'}{int(sv_raw[1:]):02d}"
                block = "".join(line.ljust(80)[:80] for line in lines[i : i + n_lines])
                rows += _values(sv, types["*"], block, epoch)
                i += n_lines
        epochs += 1

    return pl.DataFrame(
        rows,
        schema={"epoch": pl.Datetime("us", "UTC"), "sv": pl.String, "observable": pl.String, "value": pl.Float64},
        orient="row",
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "v2/obs/cgtc0920.14o",
        "v2/obs/khl20870.25o",
        "v3/obs/ASIR00ITA_R_20242810000_01D_30S_MO.rnx",
        "v3/obs/BNEU00LAO_R_20250870000_01D_30S_MO.rnx",
    ],
)
def test_parsed_values_match_raw_text(test_data_dir, relative_path):
    path = test_data_dir / relative_path
    raw = _read_raw(path, N_EPOCHS)
    assert raw["epoch"].n_unique() == N_EPOCHS
    parsed = read_rinex_obs(path)[0]
    parsed = parsed.filter(pl.col("epoch") <= raw["epoch"].max()).drop_nulls("value")

    keys = ["epoch", "sv", "observable"]
    assert parsed.height == raw.height
    joined = raw.join(parsed, on=keys, how="full", suffix="_parsed", coalesce=True)
    assert joined["value"].null_count() == 0 and joined["value_parsed"].null_count() == 0
    assert (joined["value"] - joined["value_parsed"]).abs().max() < 1e-6


def test_compressed_files_parse_like_plain_rinex(obs_v3_file, obs_v3_hatanaka_compressed_file, obs_v3_gzip_file):
    plain = read_rinex_obs(obs_v3_file)[0].sort("epoch", "sv", "observable")
    for path in (obs_v3_hatanaka_compressed_file, obs_v3_gzip_file):
        assert read_rinex_obs(path)[0].sort("epoch", "sv", "observable").equals(plain)
