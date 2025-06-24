use pyo3::prelude::*;
use rinex::prelude::*;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use std::path::Path;

#[pyfunction]
fn read_rinex_obs(path: &str) -> PyResult<(PyDataFrame, (f64, f64, f64))> {
    let rinex = Rinex::from_file(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("RINEX error: {}", e)))?;

    if !rinex.is_observation_rinex() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "This is not a RINEX Observation file",
        ));
    }

    // Extract approximate rx coordinates (ECEF)
    let (x, y, z) = rinex.header.rx_position.unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let mut epochs = Vec::new();
    let mut prns = Vec::new();
    let mut codes = Vec::new();
    let mut values = Vec::new();
    // let mut lli_flags = Vec::new();

    // Access the record containing observation data
    match &rinex.record {
        Record::ObsRecord(obs_data) => {
            for (obs_key, observations) in obs_data.iter() {
                for signal in &observations.signals {
                    epochs.push(obs_key.epoch.to_string());
                    prns.push(signal.sv.to_string());
                    codes.push(signal.observable.to_string());
                    values.push(signal.value);
                    // lli_flags.push(signal.lli.map(|f| f.bits() as i32));
                }
            }
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "File does not contain observation data",
            ));
        }
    }

    let df = df![
        "epoch" => &epochs,
        "sv" => &prns,
        "observable" => &codes,
        "value" => &values,
        // "lli" => &lli_flags,
    ]
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((PyDataFrame(df), (x, y, z)))
}

#[pyfunction]
fn read_rinex_nav(path: &str) -> PyResult<PyDataFrame> {
    let rinex = Rinex::from_file(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("RINEX error: {}", e)))?;

    if !rinex.is_navigation_rinex() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "This is not a RINEX Navigation file",
        ));
    }

    let mut prns = Vec::new();
    let mut epochs = Vec::new();
    let mut params = Vec::new();
    let mut values = Vec::new();

    for (nav_key, ephemeris) in rinex.nav_ephemeris_frames_iter() {
        let sv_str = nav_key.sv.to_string();
        let epoch_str = nav_key.epoch.to_string();

        // Clock params
        add_row(&mut prns, &mut epochs, &mut params, &mut values,
                &sv_str, &epoch_str, "clock_bias", ephemeris.clock_bias);
        add_row(&mut prns, &mut epochs, &mut params, &mut values,
                &sv_str, &epoch_str, "clock_drift", ephemeris.clock_drift);
        add_row(&mut prns, &mut epochs, &mut params, &mut values,
                &sv_str, &epoch_str, "clock_drift_rate", ephemeris.clock_drift_rate);

        // Orbit params
        let orbit_params = [
            ("sqrt_a", "sqrt_a"), 
            ("eccentricity", "eccentricity"),
            ("inclination", "inclination"),
            ("raan", "raan"),
            ("arg_perigee", "arg_perigee"),
            ("mean_anomaly", "mean_anomaly"),
            ("toe", "toe"),
            ("iode", "iode"),
            ("crc", "crc"),
            ("crs", "crs"),
            ("cuc", "cuc"),
            ("cus", "cus"),
            ("cic", "cic"),
            ("cis", "cis"),
        ];

        for (key, param_name) in orbit_params.iter() {
            if let Some(item) = ephemeris.orbits.get(*key) {
                let val = item.as_f64();
                add_row(&mut prns, &mut epochs, &mut params, &mut values,
                        &sv_str, &epoch_str, param_name, val);
            }
        }
    }

    let df = df![
        "sv" => &prns,
        "epoch" => &epochs,
        "param" => &params,
        "value" => &values,
    ]
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

fn add_row(
    prns: &mut Vec<String>,
    epochs: &mut Vec<String>,
    params: &mut Vec<String>,
    values: &mut Vec<f64>,
    sv: &str,
    epoch: &str,
    param: &str,
    value: f64,
) {
    prns.push(sv.to_string());
    epochs.push(epoch.to_string());
    params.push(param.to_string());
    values.push(value);
}

#[pymodule]
fn libtecrs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(read_rinex_nav, m)?)?;
    Ok(())
}
