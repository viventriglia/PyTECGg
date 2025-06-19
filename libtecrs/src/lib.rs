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

    for (epoch, (_, sv_map)) in rinex.obs().unwrap().epoch().iter() {
        for (sv, observations) in sv_map {
            for (code, obs) in observations {
                epochs.push(epoch.to_datetime().to_string());
                prns.push(format!("{:?}", sv));
                codes.push(code.to_string());
                values.push(obs.obs);
            }
        }
    }

    let df = df![
        "epoch" => &epochs,
        "sv" => &prns,
        "observable" => &codes,
        "value" => &values,
    ]
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

#[pymodule]
fn libtecrs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_rinex_obs_to_polars, m)?)?;
    Ok(())
}
