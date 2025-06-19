use rinex::prelude::*;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <rinex_v2_file> <rinex_v3_file>", args[0]);
        std::process::exit(1);
    }

    let (v2_path, v3_path) = (&args[1], &args[2]);

    // RINEX v2
    let now_v2 = Instant::now();
    let rinex_v2 = Rinex::from_file(v2_path).expect("Failed to read RINEX v2");
    let elapsed_v2 = now_v2.elapsed();

    // RINEX v3
    let now_v3 = Instant::now();
    let rinex_v3 = Rinex::from_file(v3_path).expect("Failed to read RINEX v3");
    let elapsed_v3 = now_v3.elapsed();

    println!("=== RINEX v2 ===");
    print_rinex_info(v2_path, &rinex_v2, elapsed_v2);

    println!("\n=== RINEX v3 ===");
    print_rinex_info(v3_path, &rinex_v3, elapsed_v3);
}

fn print_rinex_info(path: &str, rinex: &Rinex, elapsed: std::time::Duration) {
    let version = rinex.header.version;
    let file_type = if rinex.is_observation_rinex() {
        "Observation"
    } else if rinex.is_navigation_rinex() {
        "Navigation"
    } else if rinex.is_meteo_rinex() {
        "Meteo"
    } else {
        "Unknown"
    };

    println!("File: {}", path);
    println!("Version: {}", version);
    println!("Type: {}", file_type);
    println!("Parsed in: {:.3?}", elapsed);
}