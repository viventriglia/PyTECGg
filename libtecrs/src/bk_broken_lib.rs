use pyo3::prelude::*;
use rinex::Rinex;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;

#[pyfunction]
fn rinex_to_df(py: Python, input_path: &str) -> PyResult<PyDataFrame> {
    let rinex = Rinex::from_file(input_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    println!("Successfully parsed RINEX");

    let mut timestamps = Vec::new();
    let mut sats = Vec::new();

    if let Rinex::Observations(obs) = &rinex {
        for (epoch, sv_obs) in obs.iter() {
            for sv in sv_obs.keys() {
                timestamps.push(epoch.to_string());
                sats.push(sv.to_string());
            }
        }
    }

    let epoch_series = Series::new("epoch".into(), timestamps);
    let sat_series = Series::new("satellite".into(), sats);
    
    let df = DataFrame::new(vec![epoch_series.into(), sat_series.into()])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let py_df = PyDataFrame::new(py, df)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(py_df)
}

#[pymodule]
fn libtecrs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rinex_to_df, m)?)?;
    Ok(())
}
