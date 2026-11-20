//! Whole generations run headless: the physics, sensors, brains, scoring and
//! breeding together, as the windowed lab runs them.

use bevy::math::Vec2;
use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig, Task, Terrain};
use symbios_robot_lab::headless::HeadlessLab;
use symbios_robot_lab::simulation::brain::RobotBrain;
use symbios_robot_lab::simulation::hall_of_fame::{Showcase, ShowcaseRequest};
use symbios_robot_lab::simulation::lifecycle::SpawnRequest;
use symbios_robot_lab::simulation::spawn::SHOWCASE_INDEX;
use symbios_robot_lab::simulation::target::WalkTarget;
use symbios_robot_lab::simulation::telemetry::GenerationSummary;
use symbios_robot_lab::simulation::terrain::{Ground, ground_seed};

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

/// Protecting new bodies (body-only and brain-only children, youth as an
/// objective) runs whole generations and repeats from its seed.
#[test]
fn protecting_new_bodies_runs_whole_generations() {
    let config = || LabConfig {
        protect_new_bodies: true,
        ..small_config(11)
    };
    let history = run(config(), 3);
    assert_eq!(history.len(), 3);
    for summary in &history {
        assert_eq!(summary.evaluated, 12);
        assert!((1..=summary.pool_size).contains(&summary.distinct_bodies));
    }
    assert_eq!(history, run(config(), 3));
}

/// Robots walk rough ground laid afresh each generation, and a seed repeats
/// the run, grounds included.
#[test]
fn rough_ground_is_laid_afresh_each_generation() {
    let config = || LabConfig {
        terrain: Terrain::Rough,
        ..small_config(13)
    };
    let mut lab = HeadlessLab::new(config());
    let history = lab.run(2, |_| {}).expect("finishes").history.clone();
    for summary in &history {
        assert_eq!(summary.evaluated, 12);
        assert!(
            summary.exploded < 12,
            "every robot exploded on rough ground"
        );
    }
    let ground = lab.app_mut().world().resource::<Ground>();
    let (terrain, seed) = ground.built().expect("a patch was laid");
    assert_eq!(terrain, Terrain::Rough);
    assert!(
        [ground_seed(13, 1), ground_seed(13, 2)].contains(&seed),
        "the ground was not laid afresh"
    );
    assert_eq!(history, run(config(), 2));
}

/// In the target task every robot gets its own target and reads its
/// bearing, with either brain, and a seed repeats the run.
#[test]
fn robots_walk_to_targets_reading_their_bearing() {
    for brain in BrainEncoding::ALL {
        let config = || LabConfig {
            task: Task::Target,
            brain,
            ..small_config(17)
        };
        let mut lab = HeadlessLab::new(config());
        let history = lab.run(2, |_| {}).expect("finishes").history.clone();
        assert!(history.iter().all(|s| s.evaluated == 12), "{brain}");
        // A couple more updates: the next generation spawns and senses.
        lab.app_mut().update();
        lab.app_mut().update();
        let world = lab.app_mut().world_mut();
        let mut robots = world.query::<(&RobotBrain, &WalkTarget)>();
        let mut targets: Vec<Vec2> = Vec::new();
        for (robot, target) in robots.iter(world) {
            let reading = &robot.sensor_inputs[target.input_offset..target.input_offset + 2];
            let norm = reading[0].powi(2) + reading[1].powi(2);
            assert!((norm - 1.0).abs() < 1e-3, "{brain}: bearing {reading:?}");
            targets.push(target.point);
        }
        assert!(
            targets.len() > 1,
            "{brain}: {} robots have targets",
            targets.len()
        );
        // Each robot's target lies off its own spawn point.
        let distinct = targets
            .iter()
            .enumerate()
            .all(|(i, a)| targets[i + 1..].iter().all(|b| a.distance(*b) > 1.0));
        assert!(distinct, "{brain}: robots share targets");
        assert_eq!(history, run(config(), 2), "{brain}");
    }
}

/// The hall keeps the run's best robot, which can be sent into a new run's
/// first generation, as a body new to that run.
#[test]
fn champions_enter_the_hall_and_return_to_evolution() {
    let mut lab = HeadlessLab::new(small_config(19));
    let best = lab.run(3, |_| {}).expect("finishes").best_all_time;
    let champion = lab
        .hall_of_fame()
        .champions()
        .first()
        .expect("a champion")
        .clone();
    assert!((champion.fitness.expect("evaluated") - best).abs() < 1e-6);

    let mut next = HeadlessLab::new(small_config(20));
    next.send_into_evolution(champion.genotype.clone());
    next.app_mut().world_mut().write_message(SpawnRequest);
    // Bevy's first update advances no time; the next ones run fixed steps.
    for _ in 0..3 {
        next.app_mut().update();
    }
    let first = &next.population().agents;
    assert_eq!(first.len(), 12);
    assert_eq!(next.population().generation, 0);
    assert!(first[0].genotype.same_body(&champion.genotype));
    assert_eq!(first[0].genotype.body_age, 0);
    assert!(
        !first[1..]
            .iter()
            .any(|a| a.genotype.same_body(&champion.genotype))
    );
}

/// A showcased champion walks on its own, unscored, and leaves evolution
/// exactly as it would have been.
#[test]
fn a_showcase_walks_without_touching_evolution() {
    let mut plain = HeadlessLab::new(small_config(21));
    let expected = plain.run(2, |_| {}).expect("finishes").history.clone();

    let mut lab = HeadlessLab::new(small_config(21));
    let champion = plain.hall_of_fame().champions()[0].genotype.clone();
    lab.app_mut()
        .world_mut()
        .write_message(ShowcaseRequest(Some(champion)));
    let history = lab.run(2, |_| {}).expect("finishes").history.clone();
    assert_eq!(history, expected, "the showcase changed the run");

    let world = lab.app_mut().world_mut();
    let showcases = world
        .query_filtered::<(), bevy::prelude::With<Showcase>>()
        .iter(world)
        .count();
    assert_eq!(showcases, 1);
    let mut brains = world.query::<&RobotBrain>();
    assert_eq!(
        brains
            .iter(world)
            .filter(|b| b.agent_index == SHOWCASE_INDEX)
            .count(),
        1
    );
    // Dismissed, it leaves.
    world.write_message(ShowcaseRequest(None));
    lab.app_mut().update();
    let world = lab.app_mut().world_mut();
    let showcases = world
        .query_filtered::<(), bevy::prelude::With<Showcase>>()
        .iter(world)
        .count();
    assert_eq!(showcases, 0);
}

/// Under leg damage every robot walks with exactly one leg broken, and a
/// seed repeats the run.
#[test]
fn leg_damage_lames_every_robot_once() {
    let config = || LabConfig {
        leg_damage: true,
        ..small_config(23)
    };
    let mut lab = HeadlessLab::new(config());
    let history = lab.run(2, |_| {}).expect("finishes").history.clone();
    assert!(history.iter().all(|s| s.evaluated == 12));
    for _ in 0..3 {
        lab.app_mut().update();
    }
    let world = lab.app_mut().world_mut();
    let mut brains = world.query::<&RobotBrain>();
    let lamed: Vec<usize> = brains
        .iter(world)
        .filter(|b| !b.legs().is_empty())
        .map(|b| b.broken_legs().len())
        .collect();
    assert!(!lamed.is_empty());
    assert!(lamed.iter().all(|&n| n == 1), "{lamed:?}");
    assert_eq!(history, run(config(), 2));
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
