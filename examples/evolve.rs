//! Runs evolution headless from the command line and prints one row per
//! generation:
//!
//! ```text
//! cargo run --release --example evolve -- --generations 30 --seed 1
//! ```
//!
//! Options (defaults in brackets):
//! - `--generations N` [20]
//! - `--seed S` [drawn from the OS and printed]
//! - `--population P` [64]
//! - `--duration SECS` simulated seconds per generation [15]
//! - `--algorithm simple|nsga2|map-elites|cvt|novelty` [nsga2]
//! - `--brain direct|hyperneat` [hyperneat]
//! - `--mutation-rate R` [0.3]
//! - `--csv PATH` also write the rows to a CSV file

use std::error::Error;
use std::fmt::Write as _;
use std::time::Instant;
use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig};
use symbios_robot_lab::headless::HeadlessLab;
use symbios_robot_lab::simulation::telemetry::GenerationSummary;

const DEFAULT_GENERATIONS: usize = 20;
const CSV_HEADER: &str =
    "generation,best_fitness,mean_fitness,pool_best_fitness,max_displacement,exploded,seconds";

struct Options {
    generations: usize,
    config: LabConfig,
    csv: Option<String>,
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        generations: DEFAULT_GENERATIONS,
        config: LabConfig::default(),
        csv: None,
    };
    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        let number_error = |e: &dyn std::fmt::Display| format!("{flag}: {e}");
        match flag.as_str() {
            "--generations" => {
                options.generations = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--seed" => options.config.seed = Some(value()?.parse().map_err(|e| number_error(&e))?),
            "--population" => {
                options.config.population_size = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--duration" => {
                options.config.generation_duration =
                    value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--mutation-rate" => {
                options.config.mutation_rate = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--algorithm" => {
                let name = value()?;
                options.config.algorithm = GeneticAlgorithm::from_cli_name(&name)
                    .ok_or(format!("unknown algorithm {name}"))?;
            }
            "--brain" => {
                let name = value()?;
                options.config.brain =
                    BrainEncoding::from_cli_name(&name).ok_or(format!("unknown brain {name}"))?;
            }
            "--csv" => options.csv = Some(value()?),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if options.config.population_size < 2 {
        return Err("--population must be at least 2".into());
    }
    Ok(options)
}

fn row(summary: &GenerationSummary, seconds: f64) -> String {
    format!(
        "{},{:.5},{:.5},{:.5},{:.3},{},{:.2}",
        summary.generation,
        summary.best_fitness,
        summary.mean_fitness,
        summary.pool_best_fitness,
        summary.max_displacement,
        summary.exploded,
        seconds
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options(std::env::args().skip(1))?;
    let mut lab = HeadlessLab::new(options.config.clone());
    let started = Instant::now();
    let mut last = started;
    let mut csv = String::from(CSV_HEADER);
    csv.push('\n');

    println!(
        "{} generations of {} robots, {} s each, {}, {} brains",
        options.generations,
        options.config.population_size,
        options.config.generation_duration,
        options.config.algorithm,
        options.config.brain
    );
    println!("{CSV_HEADER}");
    let telemetry = lab.run(options.generations, |summary| {
        let now = Instant::now();
        let line = row(summary, now.duration_since(last).as_secs_f64());
        last = now;
        println!("{line}");
        let _ = writeln!(csv, "{line}");
    })?;
    println!(
        "best fitness {:.5} in {:.1} s",
        telemetry.best_all_time,
        started.elapsed().as_secs_f64()
    );
    if let Some(seed) = lab.seed() {
        println!("seed {seed}");
    }
    if let Some(path) = options.csv {
        std::fs::write(&path, csv)?;
        println!("wrote {path}");
    }
    Ok(())
}
