use pyo3::prelude::*;
use rinex::prelude::*;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use std::path::Path;

#[pyfunction]
fn read_rinex_obs_to_polars(path: &str) -> PyResult<PyDataFrame> {
    let rinex = Rinex::from_file(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("RINEX error: {}", e)))?;

    if !rinex.is_observation_rinex() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "This is not a RINEX Observation file",
        ));
    }

let mut epochs = Vec::new();
    let mut prns = Vec::new();
    let mut codes = Vec::new();
    let mut values = Vec::new();
    let mut lli_flags = Vec::new();

    // Access the record which contains the observation data
    match &rinex.record {
        Record::ObsRecord(obs_data) => {
            for (obs_key, observations) in obs_data.iter() {
                for signal in &observations.signals {
                    epochs.push(obs_key.epoch.to_string());
                    prns.push(signal.sv.to_string());
                    codes.push(signal.observable.to_string());
                    values.push(signal.value);
                    lli_flags.push(signal.lli.map(|f| f.bits() as i32));
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
        "lli" => &lli_flags,
    ]
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

#[pymodule]
fn libtecrs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_rinex_obs_to_polars, m)?)?;
    Ok(())
}
