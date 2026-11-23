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
//! - `--algorithm simple|nsga2|map-elites|cvt|novelty|dns` [nsga2]
//! - `--descriptor gait|body|body-gait` the traits the quality-diversity
//!   algorithms spread robots over [gait]
//! - `--brain direct|hyperneat|supg` [hyperneat]
//! - `--terrain flat|rough|hilly` new ground each generation unless flat
//!   [flat]
//! - `--task forward|target` walk forward, or to a target whose bearing the
//!   brain reads [forward]
//! - `--mutation-rate R` [0.3]
//! - `--parent-share F` share of the population kept as parents [0.5]
//! - `--retests N` best parents walking again each generation [8]
//! - `--vary-body-or-brain on|off` each child varies its body or its
//!   brain, not both [on]
//! - `--protect-new-bodies on|off` morphological innovation protection,
//!   NSGA-II only [off]
//! - `--leg-damage on|off` every robot walks with one leg broken [off]
//! - `--tilt on|off` brains also sense the body's tilt [off]
//! - `--import PATH` a robot's JSON (saved from the hall of fame) to send
//!   into the first generation; repeatable
//! - `--save-best PATH` write the run's best robot as JSON at the end
//! - `--fresh N` at the end, score the hall of fame's best four robots
//!   under N fresh conditions (their recorded scores may be lucky)
//! - `--save-hall PATH` write the hall of fame as a JSON array at the end
//! - `--score PATH` instead of evolving, score saved robots (a champion, a
//!   genotype or a whole hall) under `--fresh` conditions [8] with the
//!   other options' task, terrain and leg damage, each robot as it was
//!   bred (one bred without a steering reflex walks to targets without
//!   one); repeatable
//! - `--csv PATH` also write the rows to a CSV file
//! - `--walks PATH` with `--fresh` or `--score`, write every fresh walk as a
//!   CSV row: its score and what the score was made from (metres, speed,
//!   the gates' inputs, the motors' work) with the robot's mass, size,
//!   driven joints and feet, to judge robots by other measures than the
//!   fitness that bred them

use std::error::Error;
use std::fmt::Write as _;
use std::time::Instant;
use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig, Task, Terrain};
use symbios_robot_lab::core::descriptor::QdDescriptor;
use symbios_robot_lab::headless::{FreshWalk, HeadlessLab, condition_means, fresh_walks};
use symbios_robot_lab::simulation::hall_of_fame::Champion;
use symbios_robot_lab::simulation::telemetry::GenerationSummary;

const DEFAULT_GENERATIONS: usize = 20;
const CSV_HEADER: &str = "generation,best_fitness,mean_fitness,pool_best_fitness,\
max_displacement,exploded,distinct_bodies,oldest_body,seconds";

struct Options {
    generations: usize,
    config: LabConfig,
    csv: Option<String>,
    imports: Vec<String>,
    save_best: Option<String>,
    fresh: usize,
    save_hall: Option<String>,
    score: Vec<String>,
    walks: Option<String>,
}

/// The header of the `--walks` file.
const WALKS_HEADER: &str = "robot,believed,steps,condition,fitness,exploded,motors,feet,size,\
length,mass,metres,velocity,command_turn_hz,up_stddev,body_jerk,foot_transitions,bounce,height,\
duty,intensity,motor_work,hinge_travel,slider_travel";

/// One robot's fresh walks as rows of the `--walks` file: the walk's score
/// and what it was made from, with the robot's recorded score and its
/// body's derivation steps, driven joints, feet, size and length.
fn walk_rows(out: &mut String, robot: &str, believed: f32, steps: usize, walks: &[FreshWalk]) {
    for walk in walks {
        let t = &walk.trajectory;
        let (feet, size, length) = walk
            .body
            .map_or((0, 0.0, 0.0), |b| (b.feet, b.size, b.length));
        let _ = writeln!(
            out,
            "{robot},{believed},{steps},{},{},{},{},{feet},{size},{length},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            walk.condition,
            walk.fitness,
            u8::from(walk.exploded),
            walk.motors,
            t.mass,
            t.forward_displacement,
            t.avg_velocity,
            t.command_turn_hz,
            t.up_alignment_stddev,
            t.body_jerk,
            t.foot_transitions_per_sensor,
            t.com_bounce,
            t.avg_height,
            t.support_duty_factor,
            t.action_intensity,
            t.mechanical_work,
            t.hinge_travel,
            t.slider_travel,
        );
    }
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        generations: DEFAULT_GENERATIONS,
        config: LabConfig::default(),
        csv: None,
        imports: Vec::new(),
        save_best: None,
        fresh: 0,
        save_hall: None,
        score: Vec::new(),
        walks: None,
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
            "--parent-share" => {
                options.config.parent_share = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--retests" => {
                options.config.retests = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--fresh" => options.fresh = value()?.parse().map_err(|e| number_error(&e))?,
            "--mutation-rate" => {
                options.config.mutation_rate = value()?.parse().map_err(|e| number_error(&e))?;
            }
            "--algorithm" => {
                let name = value()?;
                options.config.algorithm = GeneticAlgorithm::from_cli_name(&name)
                    .ok_or(format!("unknown algorithm {name}"))?;
            }
            "--task" => {
                let name = value()?;
                options.config.task =
                    Task::from_cli_name(&name).ok_or(format!("unknown task {name}"))?;
            }
            "--terrain" => {
                let name = value()?;
                options.config.terrain =
                    Terrain::from_cli_name(&name).ok_or(format!("unknown terrain {name}"))?;
            }
            "--descriptor" => {
                let name = value()?;
                options.config.descriptor = QdDescriptor::from_cli_name(&name)
                    .ok_or(format!("unknown descriptor {name}"))?;
            }
            "--brain" => {
                let name = value()?;
                options.config.brain =
                    BrainEncoding::from_cli_name(&name).ok_or(format!("unknown brain {name}"))?;
            }
            "--vary-body-or-brain" => {
                options.config.vary_body_or_brain = switch(&flag, &value()?)?;
            }
            "--protect-new-bodies" => {
                options.config.protect_new_bodies = switch(&flag, &value()?)?;
            }
            "--tilt" => options.config.tilt_sense = switch(&flag, &value()?)?,
            "--leg-damage" => {
                options.config.leg_damage = switch(&flag, &value()?)?;
            }
            "--import" => options.imports.push(value()?),
            "--save-hall" => options.save_hall = Some(value()?),
            "--score" => options.score.push(value()?),
            "--walks" => options.walks = Some(value()?),
            "--save-best" => options.save_best = Some(value()?),
            "--csv" => options.csv = Some(value()?),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if options.config.population_size < 2 {
        return Err("--population must be at least 2".into());
    }
    if options.walks.is_some() && options.fresh == 0 && options.score.is_empty() {
        return Err("--walks writes fresh walks: give --fresh N or --score PATH with it".into());
    }
    Ok(options)
}

/// An on/off option's value.
fn switch(flag: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!("{flag} takes on or off, not {value}")),
    }
}

fn row(summary: &GenerationSummary, seconds: f64) -> String {
    format!(
        "{},{:.5},{:.5},{:.5},{:.3},{},{},{},{:.2}",
        summary.generation,
        summary.best_fitness,
        summary.mean_fitness,
        summary.pool_best_fitness,
        summary.max_displacement,
        summary.exploded,
        summary.distinct_bodies,
        summary.oldest_body,
        seconds
    )
}

/// Scores saved robots under fresh conditions without evolving.
fn score(options: &Options) -> Result<(), Box<dyn Error>> {
    let conditions = if options.fresh > 0 { options.fresh } else { 8 };
    let mut rows = String::from(WALKS_HEADER);
    rows.push('\n');
    for path in &options.score {
        let robots = Champion::all_from_json(&std::fs::read_to_string(path)?)
            .map_err(|err| format!("{path}: {err}"))?;
        for (i, robot) in robots.iter().enumerate() {
            let mut genotype = robot.genotype.clone();
            // Evolution gives a robot without a steering reflex a random one
            // when the task turns to targets; a robot is scored as bred.
            if options.config.task.steers() && genotype.steering_gain == 0.0 {
                genotype.steering_gain = f32::MIN_POSITIVE;
            }
            let walks = fresh_walks(&options.config, &genotype, conditions);
            let scores = condition_means(&walks);
            let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            let worst = scores.iter().copied().fold(f32::INFINITY, f32::min);
            println!(
                "score,{path},{i},{},{mean:.5},{worst:.5},{}",
                robot.fitness.unwrap_or(0.0),
                robot.walks
            );
            walk_rows(
                &mut rows,
                &format!("{path}#{i}"),
                robot.fitness.unwrap_or(0.0),
                genotype.lsystem_iterations,
                &walks,
            );
        }
    }
    if let Some(path) = &options.walks {
        std::fs::write(path, rows)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options(std::env::args().skip(1))?;
    if !options.score.is_empty() {
        return score(&options);
    }
    let mut lab = HeadlessLab::new(options.config.clone());
    for path in &options.imports {
        let champion = Champion::from_json(&std::fs::read_to_string(path)?)
            .map_err(|err| format!("{path}: {err}"))?;
        lab.send_into_evolution(champion.genotype);
        println!("imported {path}");
    }
    let started = Instant::now();
    let mut last = started;
    let mut csv = String::from(CSV_HEADER);
    csv.push('\n');

    println!(
        "{} generations of {} robots, {} s each, {}{}, {} brains, {} ground, {}",
        options.generations,
        options.config.population_size,
        options.config.generation_duration,
        options.config.algorithm,
        if options.config.protects_new_bodies() {
            " protecting new bodies"
        } else {
            ""
        },
        options.config.brain,
        options.config.terrain,
        options.config.task
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
    if options.fresh > 0 {
        let champions: Vec<_> = lab
            .hall_of_fame()
            .champions()
            .iter()
            .filter(|c| c.fitness.is_some())
            .take(4)
            .cloned()
            .collect();
        let (mut believed, mut fresh) = (Vec::new(), Vec::new());
        let mut rows = String::from(WALKS_HEADER);
        rows.push('\n');
        for (rank, champion) in champions.iter().enumerate() {
            let walks = fresh_walks(&options.config, &champion.genotype, options.fresh);
            let scores = condition_means(&walks);
            walk_rows(
                &mut rows,
                &format!("hall#{rank}"),
                champion.fitness.unwrap_or(0.0),
                champion.genotype.lsystem_iterations,
                &walks,
            );
            let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            let worst = scores.iter().copied().fold(f32::INFINITY, f32::min);
            println!(
                "fresh: {} -> {mean:.5} over {} conditions (worst {worst:.5})",
                champion.label(),
                scores.len()
            );
            believed.push(champion.fitness.unwrap_or(0.0));
            fresh.push(mean);
        }
        let mean = |xs: &[f32]| xs.iter().sum::<f32>() / xs.len().max(1) as f32;
        println!(
            "fresh score of the hall's best {}: {:.5} (believed {:.5})",
            champions.len(),
            mean(&fresh),
            mean(&believed)
        );
        if let Some(path) = &options.walks {
            std::fs::write(path, rows)?;
        }
    }
    if let Some(path) = options.save_hall {
        std::fs::write(&path, lab.hall_of_fame().to_json())?;
        println!("wrote the hall of fame to {path}");
    }
    if let Some(path) = options.save_best {
        match lab.hall_of_fame().champions().first() {
            Some(best) => {
                std::fs::write(&path, best.to_json())?;
                println!("wrote the best robot ({}) to {path}", best.label());
            }
            None => println!("no robot walked: nothing written to {path}"),
        }
    }
    Ok(())
}
