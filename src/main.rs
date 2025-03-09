#![allow(unused)]

use std::path::PathBuf;
use clap::Parser;
use config_file::FromConfigFile;
use rustcar2::cars;
use serde::Deserialize;


/// Command line argument parser.
#[derive(Parser, Debug)]
#[command(about = "Solve the Barto and Sutton Car Rental Problem", long_about = None)]
pub struct Args {
    /// Path to RustCar configuration TOML file.
    config_path: Option<PathBuf>
}

/// Hold information read form TOML configuration file.
#[derive(Deserialize, Debug)]
pub struct CarConfig {
    pub max1: u8,
    pub rent_mean1: f32,
    pub return_mean1: f32,
    pub max2: u8,
    pub rent_mean2: f32,
    pub return_mean2: f32,
    pub max_move: u8
}


fn main() {
    let args = Args::parse();
    let config_path = args.config_path
        .expect("No file path provided.");
    println!("Reading config file: {}", config_path.to_str()
        .expect("Involid file path."));
    let config = CarConfig::from_config_file(config_path)
        .expect("Unable to read configuration file.");
    println!("{:?}", config);

    let cprobs = cars::CarProbs::new(
        config.max1, config.rent_mean1, config.return_mean1,
        config.max2, config.rent_mean2, config.return_mean2,
        config.max_move);
}
