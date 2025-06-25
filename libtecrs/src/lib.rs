use pyo3::prelude::*;
use rinex::prelude::*;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use std::path::Path;
use std::collections::{BTreeSet, BTreeMap};

#[pyfunction]
fn read_rinex_obs(path: &str) -> PyResult<(PyDataFrame, (f64, f64, f64))> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("File not found: {}", path.display())
        ));
    }

    let rinex = Rinex::from_file(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("RINEX parsing error: {}", e)
        ))?;

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
fn read_rinex_nav(path: &str) -> PyResult<BTreeMap<String, PyDataFrame>> {
    let rinex = Rinex::from_file(Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("RINEX error: {}", e)))?;

    if !rinex.is_navigation_rinex() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "This is not a RINEX Navigation file",
        ));
    }

    // Mappa per costellazione -> DataFrame
    let mut constellation_data: BTreeMap<String, Vec<BTreeMap<String, f64>>> = BTreeMap::new();
    let mut constellation_times: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut constellation_svs: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (nav_key, ephemeris) in rinex.nav_ephemeris_frames_iter() {
        let constellation = match nav_key.sv.constellation {
            Constellation::GPS => "GPS",
            Constellation::Glonass => "GLONASS",
            Constellation::Galileo => "Galileo",
            Constellation::BeiDou => "BeiDou",
            Constellation::QZSS => "QZSS",
            Constellation::IRNSS => "IRNSS",
            Constellation::SBAS => "SBAS",
            _ => "Unknown",
        }.to_string();

        let sv_id = nav_key.sv.prn.to_string(); // Rimuove il prefisso della costellazione
        let epoch_str = nav_key.epoch.to_string();

        // Crea una mappa per tutti i parametri
        let mut params = BTreeMap::new();

        // Aggiungi parametri di clock
        params.insert("clock_bias".to_string(), ephemeris.clock_bias);
        params.insert("clock_drift".to_string(), ephemeris.clock_drift);
        params.insert("clock_drift_rate".to_string(), ephemeris.clock_drift_rate);

        // Aggiungi tutti i parametri orbitali disponibili
        for (key, value) in &ephemeris.orbits {
            params.insert(key.to_string(), value.as_f64());
        }

        // Inizializza le strutture dati per questa costellazione se non esistono
        constellation_data.entry(constellation.clone())
            .or_insert_with(Vec::new)
            .push(params);
        constellation_times.entry(constellation.clone())
            .or_insert_with(Vec::new)
            .push(epoch_str);
        constellation_svs.entry(constellation.clone())
            .or_insert_with(Vec::new)
            .push(sv_id);
    }

    // Crea i DataFrame per ogni costellazione
    let mut result = BTreeMap::new();
    
    for (constellation, data) in constellation_data {
        let times = &constellation_times[&constellation];
        let svs = &constellation_svs[&constellation];
        
        // Raccogli tutti i nomi dei parametri univoci
        let mut all_params = BTreeSet::new();
        for params in &data {
            for param_name in params.keys() {
                all_params.insert(param_name.clone());
            }
        }

        // Crea una serie per ogni parametro
        let mut series_map: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        for param in &all_params {
            series_map.insert(param.clone(), Vec::new());
        }

        // Popola le serie
        for params in &data {
            for param in &all_params {
                let value = params.get(param).copied();
                series_map.get_mut(param).unwrap().push(value);
            }
        }

        // Crea il DataFrame
        let mut df_builder = df! {
            "epoch" => times,
            "sv" => svs,
        }.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Aggiungi tutte le colonne dei parametri
        for (param_name, values) in series_map {
            let mut series: Series = values.into_iter()
                .map(|opt| opt.map(|v| v as f64))
                .collect::<Float64Chunked>()
                .into_series();
            series.rename(param_name.into());
            df_builder.with_column(series)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }

        // Imposta l'indice multi-livello (epoch, sv)
        df_builder = df_builder
            .lazy()
            .with_row_index("row_id", None)
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        result.insert(constellation, PyDataFrame(df_builder));
    }

    Ok(result)
}

#[pymodule]
fn pytecal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(read_rinex_nav, m)?)?;
    Ok(())
}
