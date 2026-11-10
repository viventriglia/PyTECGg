import polars as pl

from pytecgg.parsing import read_rinex_nav


def test_blank_field_in_first_record_does_not_shift_column(nav_v3_file, tmp_path):
    """A field missing from the first record of a constellation must be null
    for that record only, never shift the column onto other records."""
    original = read_rinex_nav(nav_v3_file)["GPS"]
    first = original.sort("epoch", "sv").row(0, named=True)
    tag = f"G{int(first['sv']):02d} {first['epoch'].strftime('%Y %m %d %H %M %S')}"

    lines = open(nav_v3_file).read().split("\n")
    i = next(k for k, line in enumerate(lines) if line.startswith(tag))
    # toe is the first field of the 4th line of a RINEX 3 GPS record
    lines[i + 3] = lines[i + 3][:4] + " " * 19 + lines[i + 3][23:]
    patched = tmp_path / "nav.rnx"
    patched.write_text("\n".join(lines))

    changed = (
        original.select("epoch", "sv", "toe")
        .join(read_rinex_nav(patched)["GPS"].select("epoch", "sv", "toe"), on=["epoch", "sv"])
        .filter(pl.col("toe").ne_missing(pl.col("toe_right")))
    )
    assert changed.select("epoch", "sv").rows() == [(first["epoch"], first["sv"])]
    assert changed["toe_right"][0] is None


def test_zero_values_are_kept(nav_v3_file):
    """Zero is a real value: GLONASS R11/R15 broadcast on channel 0 and must
    keep it, or they are dropped from TEC."""
    glonass = read_rinex_nav(nav_v3_file)["GLONASS"]
    channels = glonass.filter(pl.col("sv").is_in(["11", "15"]))["channel"]
    assert channels.len() > 0
    assert channels.null_count() == 0
    assert (channels == 0).all()
