#![allow(unused)]

use std::{future::poll_fn, path::PathBuf};
use clap::{Parser, Subcommand};
use config_file::FromConfigFile;
use serde::Deserialize;

use rustcar2::{cars::CarProbs, policy, solver::State};


/// Command line argument parser.
#[derive(Parser, Debug)]
#[command(about = "Solve the Barto and Sutton Car Rental Problem", long_about = None)]
pub struct Args {
    /// Path to RustCar configuration TOML file.
    config_path: PathBuf,

    #[command(subcommand)]
    command: Commands
}


#[derive(Subcommand, Debug)]
enum Commands {
    /// Print rental and return probabilities.
    Probs,
    /// Calculate expected reward for a state.
    Reward {n1: u8, n2: u8},
    /// Solve for optimal policy
    Trace {s1_n1: u8, s1_n2: u8, s2_n1: u8, s2_n2: u8, a: i8, xt: u32},
    Solve
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
    pub max_move: u8,
    pub gamma: f32
}


fn main() {
    let args = Args::parse();
    let cprobs = get_carprobs_from_config(&args.config_path);

    match &args.command {
        Commands::Probs => {
            cprobs.show_probs();
        }
        Commands::Reward {n1, n2} => {
            let r = cprobs.calc_value(&State {n1: *n1, n2: *n2});
            println!("Expected Reward: {:.2}", r);
        }
        Commands::Trace {s1_n1, s1_n2, s2_n1, s2_n2, a, xt  } => {
            let s1 = State { n1: *s1_n1, n2: *s1_n2 };
            let s2 = State { n1: *s2_n1, n2: *s2_n2 };
            let (r, prob, oprobs) = cprobs.calc_reward_prob(&s1, &s2, *a, *xt);
            for oc in oprobs {
                println!("{:?}", oc);
            }
        }
        Commands::Solve => {println!("Solve the car rental problem.???!!")}
    }

    println!("Initializing Policy.");
    let cpolicy = rustcar2::policy::Policy::new(
        cprobs.max1, cprobs.max2, cprobs.max_move
    );
}


fn get_carprobs_from_config(config_path: &PathBuf) -> CarProbs {
    println!("Reading config file: {}", config_path.to_str()
        .expect("Involid file path."));
    let config = CarConfig::from_config_file(config_path)
        .expect("Unable to read configuration file.");
    println!("Calculating rental and return probabilities.");
    let cprobs = rustcar2::cars::CarProbs::new(
        config.max1, config.rent_mean1, config.return_mean1,
        config.max2, config.rent_mean2, config.return_mean2,
        config.max_move);
    cprobs
}
