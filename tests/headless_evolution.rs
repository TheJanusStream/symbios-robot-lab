//! Whole generations run headless: the physics, sensors, brains, scoring and
//! breeding together, as the windowed lab runs them.

use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig};
use symbios_robot_lab::headless::HeadlessLab;
use symbios_robot_lab::simulation::telemetry::GenerationSummary;

/// A small, short run: enough robots and seconds to exercise everything.
fn small_config(seed: u64) -> LabConfig {
    LabConfig {
        population_size: 12,
        generation_duration: 6.0,
        seed: Some(seed),
        ..LabConfig::default()
    }
}

fn run(config: LabConfig, generations: usize) -> Vec<GenerationSummary> {
    let mut lab = HeadlessLab::new(config);
    lab.run(generations, |_| {})
        .expect("the run finishes its generations")
        .history
        .clone()
}

#[test]
fn a_short_run_finishes_every_generation_and_robots_move() {
    let history = run(small_config(1), 3);
    assert_eq!(history.len(), 3);
    for (i, summary) in history.iter().enumerate() {
        assert_eq!(summary.generation, i);
        assert_eq!(summary.evaluated, 12, "generation {i} lost robots");
        assert!(summary.best_fitness.is_finite() && summary.best_fitness >= 0.0);
        assert!(summary.mean_fitness.is_finite());
        assert!(
            summary.exploded < 12,
            "generation {i}: every robot exploded"
        );
    }
    let furthest = history
        .iter()
        .map(|s| s.max_displacement)
        .fold(0.0, f32::max);
    assert!(
        furthest > 0.05,
        "no robot moved more than 5 cm forward in three generations ({furthest} m)"
    );
}

/// A seed repeats a whole run exactly, with either brain: spawning, physics
/// (the headless app is single-threaded), sensors, brains, scoring and
/// breeding. Breeding needs `symbios` 1.5.3 or later; 1.5.2 crossed and
/// mutated L-systems in hash order, so runs diverged after generation 0.
#[test]
fn the_same_seed_repeats_a_whole_run_exactly() {
    for brain in BrainEncoding::ALL {
        let config = || LabConfig {
            brain,
            ..small_config(7)
        };
        let a = run(config(), 4);
        let b = run(config(), 4);
        assert_eq!(a.len(), 4);
        assert_eq!(a, b, "{brain} runs differ");
    }
}

/// The default brain is HyperNEAT; direct NEAT brains run whole generations
/// too.
#[test]
fn direct_brains_drive_robots_through_generations() {
    let config = LabConfig {
        brain: BrainEncoding::Direct,
        ..small_config(5)
    };
    let history = run(config, 2);
    assert_eq!(history.len(), 2);
    assert!(history.iter().all(|s| s.evaluated == 12));
    let furthest = history
        .iter()
        .map(|s| s.max_displacement)
        .fold(0.0, f32::max);
    assert!(
        furthest > 0.05,
        "no direct-brained robot moved 5 cm ({furthest} m)"
    );
}

#[test]
fn every_algorithm_completes_a_generation() {
    for algorithm in GeneticAlgorithm::ALL {
        let config = LabConfig {
            algorithm,
            population_size: 8,
            generation_duration: 3.0,
            ..small_config(3)
        };
        let history = run(config, 2);
        assert_eq!(history.len(), 2, "{algorithm} did not finish");
        assert_eq!(history[1].evaluated, 8, "{algorithm} lost robots");
    }
}
