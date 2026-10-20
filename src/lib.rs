use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rinex::prelude::*;
use polars::prelude::*;
use std::path::Path;
use std::collections::BTreeMap;

/// Constant offset between J1900 (hifitime default) and Unix Epoch (1970) in microseconds,
/// including the 19s constant offset between TAI and GPST.
/// This ensures RINEX epochs align with the "round" 00/30s grid in Polars/Unix time.
const UNIX_GPST_OFFSET_MICROS: i64 = 2_208_988_819_000_000;

/// Helper function to read a RINEX file (supports regular, compressed, and gzipped RINEX files)
fn _parse_file<P: AsRef<Path>>(path: P) -> Result<Rinex, ParsingError> {
    let path = path.as_ref();

    // 1. Try with gzip, if file has .gz extension
    if path.extension().map_or(false, |ext| ext == "gz") {
        if let Ok(rinex) = Rinex::from_gzip_file(path) {
            return Ok(rinex);
        }
        // If gzip fails, continue to try regular parsing
    }

    // 2. Try regular file parsing (works for both .rnx and .crx)
    Rinex::from_file(path)
}


/// Parses a RINEX observation file and returns the extracted observation data as a DataFrame
///
/// Parameters:
///     path (str): Path to the RINEX observation file
///
/// Returns:
///     tuple:
///         - PyDataFrame: A DataFrame with columns 'epoch', 'sv', 'observable', 'value'
///         - tuple[float, float, float]: Receiver's position in ECEF coordinates (in meters)
///         - str: RINEX version
#[pyfunction]
#[pyo3(text_signature = "(path, /)")]
fn read_rinex_obs(path: &str) -> PyResult<(PyDataFrame, (f64, f64, f64), String)> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("File not found: {}", path.display())
        ));
    }

    let rinex = _parse_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("RINEX parsing error: {}", e)
        ))?;

    if !rinex.is_observation_rinex() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not an OBS file"));
    }

    let (x, y, z) = rinex.header.rx_position.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let version = rinex.header.version.to_string();

    let est_capacity = 250_000;
    let mut epochs = Vec::with_capacity(est_capacity);
    let mut prns = Vec::with_capacity(est_capacity);
    let mut codes = Vec::with_capacity(est_capacity);
    let mut values = Vec::with_capacity(est_capacity);

    match &rinex.record {
        Record::ObsRecord(obs_data) => {
            for (obs_key, observations) in obs_data.iter() {
                // Bypass UTC leap second adjustments to preserve original GPST grid.
                let total_micros = (obs_key.epoch.to_duration_since_j1900().to_seconds() * 1_000_000.0) as i64;
                let ts = total_micros - UNIX_GPST_OFFSET_MICROS;

                for signal in &observations.signals {
                    epochs.push(ts);
                    // {:x} is always "S23"-style; Display names SBAS vehicles ("ASTRA-5B")
                    prns.push(format!("{:x}", signal.sv));
                    codes.push(signal.observable.to_string());
                    values.push(signal.value);
                }
            }
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("No obs data")),
    }

    let epoch_series = Series::new("epoch".into(), epochs)
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let df = DataFrame::new(vec![
        epoch_series.into(),
        Series::new("sv".into(), prns).into(),
        Series::new("observable".into(), codes).into(),
        Series::new("value".into(), values).into(),
    ])
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((PyDataFrame(df), (x, y, z), version))
}


/// Parses a RINEX navigation file and returns a dictionary of DataFrames,
/// one per GNSS constellation
///
/// Parameters:
///     path (str): Path to the RINEX navigation file
///
/// Returns:
///     dict[str, PyDataFrame]: A dictionary where keys are GNSS constellation names
///     (e.g., "GPS", "Galileo") and values are DataFrames containing navigation parameters
#[pyfunction]
#[pyo3(text_signature = "(path, /)")]
fn read_rinex_nav(path: &str) -> PyResult<BTreeMap<String, PyDataFrame>> {
    let path_obj = Path::new(path);
    let rinex = _parse_file(path_obj).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

    let mut storage: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
    let mut constellation_times: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    let mut constellation_svs: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (nav_key, ephemeris) in rinex.nav_ephemeris_frames_iter() {
        let constel = match nav_key.sv.constellation {
            Constellation::GPS => "GPS",
            Constellation::Glonass => "GLONASS",
            Constellation::Galileo => "GALILEO",
            Constellation::BeiDou => "BEIDOU",
            // Constellation::QZSS => "QZSS",
            // Constellation::IRNSS => "IRNSS",
            // Constellation::SBAS => "SBAS",
            // _ => "OTHER", // Uncomment to include other constellations
            _ => continue, // Skip unsupported constellations
        }.to_string();
        
        let (y, m, d, hh, mm, ss, ns) = nav_key.epoch.to_gregorian(nav_key.epoch.time_scale);
        let forced_epoch = Epoch::from_gregorian(y, m, d, hh, mm, ss, ns, TimeScale::GPST);
        let total_micros = (forced_epoch.to_duration_since_j1900().to_seconds() * 1_000_000.0) as i64;
        let ts = total_micros - UNIX_GPST_OFFSET_MICROS;

        let times = constellation_times.entry(constel.clone()).or_default();
        times.push(ts);
        let current_len = times.len();
        constellation_svs.entry(constel.clone()).or_default().push(nav_key.sv.prn.to_string());

        let params_map = storage.entry(constel.clone()).or_default();
        // A column first seen at this record starts with nulls for all earlier records
        let mut push = |key: String, value: f64| {
            params_map.entry(key).or_insert_with(|| vec![None; current_len - 1]).push(Some(value));
        };

        push("clock_bias".into(), ephemeris.clock_bias);
        push("clock_drift".into(), ephemeris.clock_drift);
        push("clock_drift_rate".into(), ephemeris.clock_drift_rate);

        for (key, value) in &ephemeris.orbits {
            push(key.to_string(), value.as_f64());
        }

        for vec in params_map.values_mut() {
            if vec.len() < current_len {
                vec.push(None);
            }
        }
    }

    let mut result = BTreeMap::new();
    for (constel, columns) in storage {
        let times = constellation_times.remove(&constel).unwrap();
        let svs = constellation_svs.remove(&constel).unwrap();

        let epoch_series = Series::new("epoch".into(), times)
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let mut df = DataFrame::new(vec![
            epoch_series.into(),
            Series::new("sv".into(), svs).into(),
        ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        for (name, values) in columns {
            let mut final_values = values;
            while final_values.len() < df.height() {
                final_values.push(None);
            }
            let s = Series::new(name.into(), final_values);
            df.with_column(s).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }

        result.insert(constel, PyDataFrame(df));
    }

    Ok(result)
}


#[pymodule]
fn pytecgg(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(read_rinex_nav, m)?)?;
    Ok(())
}
