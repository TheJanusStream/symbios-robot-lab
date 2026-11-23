//! Whole generations run headless: the physics, sensors, brains, scoring and
//! breeding together, as the windowed lab runs them.

use bevy::math::Vec2;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig, Task, Terrain};
use symbios_robot_lab::core::genome::AgentGenotype;
use symbios_robot_lab::core::presets::{MORPHOLOGY_POOL, PLAN_NAMES};
use symbios_robot_lab::headless::{
    FRESH_CLONES, FRESH_SPOT_REACH, FreshSpots, HeadlessLab, condition_means, fresh_scores,
    fresh_walks, fresh_walks_from,
};
use symbios_robot_lab::simulation::brain::RobotBrain;
use symbios_robot_lab::simulation::hall_of_fame::{
    Entrant, LANE_SPACING, Lane, Showcase, ShowcaseRequest,
};
use symbios_robot_lab::simulation::hyperneat::fresh_cppn;
use symbios_robot_lab::simulation::lifecycle::SpawnRequest;
use symbios_robot_lab::simulation::lineage::{Lineage, RACE_LANES};
use symbios_robot_lab::simulation::physics::GRAVITY;
use symbios_robot_lab::simulation::spawn::{SHOWCASE_INDEX, showcase_index};
use symbios_robot_lab::simulation::target::{TARGET_LEGS, WalkTarget};
use symbios_robot_lab::simulation::telemetry::{GenerationSummary, Telemetry};
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
/// first generation, as a body new to that run. (No parent walks again
/// here: one that does is kept at its average, which the best single walk
/// of the run need not equal.)
#[test]
fn champions_enter_the_hall_and_return_to_evolution() {
    let mut lab = HeadlessLab::new(LabConfig {
        retests: 0,
        ..small_config(19)
    });
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

/// A race of the run's generations: their champions walk on their own, side
/// by side in lanes, unscored, and leave evolution exactly as it would have
/// been.
#[test]
fn a_race_of_generations_walks_without_touching_evolution() {
    let mut plain = HeadlessLab::new(small_config(21));
    let expected = plain.run(2, |_| {}).expect("finishes").history.clone();
    // The lineage holds generation 0's best and the latest, generation 1's.
    let lineage = plain.app_mut().world().resource::<Lineage>();
    let race: Vec<_> = lineage.race(RACE_LANES).into_iter().cloned().collect();
    let generations: Vec<usize> = race.iter().map(|m| m.generation).collect();
    assert_eq!(generations, [0, 1]);
    let entrants = race
        .iter()
        .map(|m| Entrant {
            name: format!("Generation {}", m.generation),
            genotype: m.champion.genotype.clone(),
        })
        .collect();

    let mut lab = HeadlessLab::new(small_config(21));
    lab.app_mut()
        .world_mut()
        .write_message(ShowcaseRequest(entrants));
    let history = lab.run(2, |_| {}).expect("finishes").history.clone();
    assert_eq!(history, expected, "the race changed the run");

    let world = lab.app_mut().world_mut();
    let showcases = world
        .query_filtered::<(), bevy::prelude::With<Showcase>>()
        .iter(world)
        .count();
    assert_eq!(showcases, 2);
    let mut lanes: Vec<(usize, Lane)> = world
        .query::<(&RobotBrain, &Lane)>()
        .iter(world)
        .map(|(brain, lane)| (brain.agent_index, lane.clone()))
        .collect();
    lanes.sort_by_key(|(_, lane)| lane.lane);
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes[0].0, SHOWCASE_INDEX);
    assert_eq!(lanes[1].0, showcase_index(1));
    assert_eq!(lanes[1].1.name, "Generation 1");
    let apart = lanes[0].1.start.distance(lanes[1].1.start);
    assert!((apart - LANE_SPACING).abs() < 1e-4, "{apart}");
    // Dismissed, they leave; one robot then walks alone in the first lane.
    world.write_message(ShowcaseRequest::dismiss());
    lab.app_mut().update();
    let world = lab.app_mut().world_mut();
    let showcases = world
        .query_filtered::<(), bevy::prelude::With<Showcase>>()
        .iter(world)
        .count();
    assert_eq!(showcases, 0);
    let champion = race[1].champion.genotype.clone();
    world.write_message(ShowcaseRequest::one("champion", champion));
    lab.app_mut().update();
    let world = lab.app_mut().world_mut();
    let alone: Vec<usize> = world
        .query::<(&RobotBrain, &Lane)>()
        .iter(world)
        .map(|(brain, _)| brain.agent_index)
        .collect();
    assert_eq!(alone, [SHOWCASE_INDEX]);
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

/// With fewer parents, the best of them walk again each generation: the run
/// repeats from its seed, and champions that walked again carry averages.
#[test]
fn the_best_parents_walk_again() {
    let config = || LabConfig {
        parent_share: 0.5,
        retests: 8,
        ..small_config(29)
    };
    let mut lab = HeadlessLab::new(config());
    let history = lab.run(4, |_| {}).expect("finishes").history.clone();
    assert!(history.iter().all(|s| s.evaluated == 12));
    assert!(history.iter().skip(1).all(|s| s.pool_size <= 6));
    assert!(
        lab.hall_of_fame().champions().iter().any(|c| c.walks > 1),
        "no champion walked again"
    );
    assert_eq!(history, run(config(), 4));
}

/// A run can breed from plans of its own in place of the lab's: every
/// robot then descends from one of them. Mutation never edits an axiom, and
/// of the lab's plans only the scout's axiom mounts an eye, so a run bred
/// from the scout alone holds nothing but robots with one.
#[test]
fn a_run_breeds_from_the_plans_it_is_given() {
    let scout = PLAN_NAMES
        .iter()
        .position(|name| *name == "Scout")
        .expect("a scout");
    let has_eye = |morphology: &str| {
        morphology
            .lines()
            .any(|line| line.starts_with("omega:") && line.contains(" Su"))
    };
    let lone: Vec<bool> = MORPHOLOGY_POOL.iter().map(|plan| has_eye(plan)).collect();
    assert_eq!(lone.iter().filter(|eye| **eye).count(), 1);
    assert!(lone[scout]);

    let mut lab = HeadlessLab::new(small_config(41));
    lab.breed_from([("scout.lsys".to_string(), MORPHOLOGY_POOL[scout].to_string())]);
    lab.run(2, |_| {}).expect("finishes");
    let champions = lab.hall_of_fame().champions();
    assert!(!champions.is_empty(), "a robot walked");
    assert!(champions.iter().all(|c| has_eye(&c.genotype.morphology)));
    let agents = &lab.population().agents;
    assert!(agents.iter().all(|a| has_eye(&a.genotype.morphology)));

    // Without them, the lab's three plans all turn up; and so they do when
    // the plans named first are taken back.
    let mut lab = HeadlessLab::new(small_config(41));
    lab.run(1, |_| {}).expect("finishes");
    let agents = &lab.population().agents;
    assert!(agents.iter().any(|a| !has_eye(&a.genotype.morphology)));

    let mut lab = HeadlessLab::new(small_config(41));
    lab.breed_from([("scout.lsys".to_string(), MORPHOLOGY_POOL[scout].to_string())]);
    lab.breed_from([]);
    lab.run(1, |_| {}).expect("finishes");
    let agents = &lab.population().agents;
    assert!(agents.iter().any(|a| !has_eye(&a.genotype.morphology)));
}

/// Fresh scores: a robot walks under new conditions, one score per
/// condition, never below zero.
#[test]
fn fresh_scores_walk_a_robot_under_new_conditions() {
    let mut lab = HeadlessLab::new(small_config(31));
    lab.run(2, |_| {}).expect("finishes");
    let champion = lab.hall_of_fame().champions()[0].clone();
    let scores = fresh_scores(&small_config(31), &champion.genotype, 3);
    assert_eq!(scores.len(), 3);
    assert!(scores.iter().all(|s| s.is_finite() && *s >= 0.0));
    assert_eq!(
        scores,
        fresh_scores(&small_config(31), &champion.genotype, 3)
    );
}

/// A fresh condition sets its robots down somewhere else, within reach of
/// where evolution's stood, and the same condition always on the same
/// spots.
#[test]
fn a_fresh_condition_moves_the_spots() {
    let shifts: Vec<Vec2> = (0..8).map(|k| FreshSpots::Moved.shift(k)).collect();
    for (k, shift) in shifts.iter().enumerate() {
        assert!(shift.abs().max_element() <= FRESH_SPOT_REACH, "{shift}");
        assert_eq!(*shift, FreshSpots::Moved.shift(k as u64));
        assert_eq!(FreshSpots::Same.shift(k as u64), Vec2::ZERO);
        // No two conditions within a metre of each other.
        for other in &shifts[..k] {
            assert!(shift.distance(*other) > 1.0, "{shift} and {other}");
        }
    }
}

/// On the flat floor a fresh condition changes only where the robots
/// start, and that makes it a condition of its own: a champion walks a
/// different walk in each. Set down on the same spots, as fresh walks were
/// before, every condition repeats the first, and eight conditions were
/// two walks.
#[test]
fn every_fresh_condition_is_another_walk_on_the_flat_floor() {
    let mut lab = HeadlessLab::new(small_config(31));
    lab.run(3, |_| {}).expect("finishes");
    let champion = lab.hall_of_fame().champions()[0].clone();
    let config = small_config(31);
    // Each condition's walks: how far each clone got.
    let metres = |spots: FreshSpots| -> Vec<Vec<f32>> {
        let walks = fresh_walks_from(&config, &champion.genotype, 3, spots);
        (0..3)
            .map(|k| {
                let of_k = walks.iter().filter(|w| w.condition == k);
                of_k.map(|w| w.trajectory.forward_displacement).collect()
            })
            .collect()
    };
    let same = metres(FreshSpots::Same);
    assert_eq!(same[0].len(), FRESH_CLONES);
    assert_eq!(same[1], same[0]);
    assert_eq!(same[2], same[0]);

    let moved = metres(FreshSpots::Moved);
    assert_ne!(moved[0], moved[1]);
    assert_ne!(moved[1], moved[2]);
    assert_ne!(moved[0], moved[2]);
    // Moved is what the fresh scores walk.
    let walks = fresh_walks(&config, &champion.genotype, 3);
    let first: Vec<f32> = walks
        .iter()
        .filter(|w| w.condition == 0)
        .map(|w| w.trajectory.forward_displacement)
        .collect();
    assert_eq!(first, moved[0]);
}

/// A joint whose limit gives no effort has no motor. A skink whose joints
/// are all so goes limp, and its motors do no work, where the same robot
/// with its motors does. (The physics reads a motor's limit of 0 as "no
/// limit": handed on, the 0 made every such joint an unlimited one.)
#[test]
fn a_joint_without_effort_has_no_motor() {
    let skink = MORPHOLOGY_POOL[0];
    assert!(skink.contains("Jl(-dl, dh, 4000, 5)"));
    let strong = AgentGenotype::new(
        skink.to_string(),
        1,
        BrainEncoding::HyperNeat,
        fresh_cppn(&mut Pcg64::seed_from_u64(5)),
    );
    let mut limp = strong.clone();
    limp.morphology = limp.morphology.replace("4000", "0");
    assert_ne!(limp.morphology, strong.morphology);

    let config = small_config(31);
    let work = |genotype: &AgentGenotype| -> Vec<f32> {
        let walks = fresh_walks(&config, genotype, 1);
        assert_eq!(walks.len(), FRESH_CLONES);
        assert!(walks.iter().all(|w| w.motors == 6 && !w.exploded));
        walks.iter().map(|w| w.trajectory.mechanical_work).collect()
    };
    let with_motors = work(&strong);
    assert!(with_motors.iter().all(|w| *w > 0.0), "{with_motors:?}");
    let without = work(&limp);
    assert!(without.iter().all(|w| *w == 0.0), "{without:?}");
}

/// A walk measures what the fitness and the economy objective are made of:
/// the champions' commands turn round, their motors do work wherever they
/// get anywhere, and the work is a believable one for the weight moved.
/// The fresh scores are these walks' fitness, averaged per condition.
#[test]
fn a_walk_counts_command_turns_and_the_motors_work() {
    let mut lab = HeadlessLab::new(small_config(31));
    lab.run(3, |_| {}).expect("finishes");
    let champions = lab.hall_of_fame().champions().to_vec();
    assert!(!champions.is_empty(), "a robot walked");
    let (mut walked, mut turned) = (0, 0);
    for champion in &champions {
        let walks = fresh_walks(&small_config(31), &champion.genotype, 2);
        assert_eq!(walks.len(), 2 * FRESH_CLONES);
        assert_eq!(
            condition_means(&walks),
            fresh_scores(&small_config(31), &champion.genotype, 2)
        );
        for walk in &walks {
            let t = &walk.trajectory;
            assert!(walk.motors > 4, "{} motors", walk.motors);
            assert!(walk.body.is_some_and(|body| body.size > 0.5));
            // Brains run 32 times a second: no command turns faster.
            assert!((0.0..=16.0).contains(&t.command_turn_hz));
            if walk.exploded || t.forward_displacement < 0.2 {
                continue;
            }
            walked += 1;
            turned += usize::from(t.command_turn_hz > 0.0);
            let cost = t.mechanical_work / (t.mass * GRAVITY * t.forward_displacement);
            assert!(
                (0.01..1000.0).contains(&cost),
                "{} J to move {} kg {} m",
                t.mechanical_work,
                t.mass,
                t.forward_displacement
            );
        }
    }
    assert!(
        walked > 0,
        "no champion got anywhere under fresh conditions"
    );
    assert!(
        turned > 0,
        "no walking champion's command ever turned round"
    );
}

/// By the end of a target-task generation every robot has walked through
/// all its legs, each target on the other side of the last, with a steering
/// reflex of its own.
#[test]
fn a_target_task_life_has_legs_on_both_sides() {
    let mut lab = HeadlessLab::new(LabConfig {
        task: Task::Target,
        ..small_config(37)
    });
    lab.app_mut()
        .world_mut()
        .resource_mut::<LabConfig>()
        .auto_restart = false;
    lab.app_mut().world_mut().write_message(SpawnRequest);
    for _ in 0..200 {
        lab.app_mut().update();
        let world = lab.app_mut().world();
        if world
            .resource::<symbios_robot_lab::simulation::clock::GenerationClock>()
            .is_finished()
        {
            break;
        }
    }
    let world = lab.app_mut().world_mut();
    let mut robots = world.query::<(&RobotBrain, &WalkTarget)>();
    let mut count = 0;
    for (robot, target) in robots.iter(world) {
        assert_eq!(target.leg, TARGET_LEGS);
        let steering = robot.steering.as_ref().expect("a steering reflex");
        assert!(steering.gain != 0.0 && steering.sides.len() == robot.outputs.len());
        // The reflex scales outputs, but they stay what a motor can do.
        assert!(
            robot.outputs.iter().all(|o| (-1.0..=1.0).contains(o)),
            "{:?}",
            robot.outputs
        );
        count += 1;
    }
    assert_eq!(count, 12);
    // The generation is over and waits (auto-advance off): the robots walk
    // on, but their scores stay as the last step left them.
    let scores = |lab: &HeadlessLab| -> Vec<f32> {
        lab.population().agents.iter().map(|a| a.fitness).collect()
    };
    let at_the_end = scores(&lab);
    for _ in 0..5 {
        lab.app_mut().update();
    }
    assert_eq!(scores(&lab), at_the_end);
}

/// A showcased robot outlives generations: in the target task it starts its
/// legs again with each one, toward new targets.
#[test]
fn a_showcased_robot_gets_new_targets_every_generation() {
    let config = LabConfig {
        task: Task::Target,
        ..small_config(47)
    };
    let mut plain = HeadlessLab::new(config.clone());
    plain.run(1, |_| {}).expect("finishes");
    let champion = plain.hall_of_fame().champions()[0].genotype.clone();

    let mut lab = HeadlessLab::new(config);
    let world = lab.app_mut().world_mut();
    world.write_message(ShowcaseRequest::one("champion", champion));
    world.write_message(SpawnRequest);
    // The showcased robot's target as each generation ends (the next may
    // already have begun, and the robot started its legs again, still heading
    // for its last target).
    let mut ends: Vec<WalkTarget> = Vec::new();
    for _ in 0..400 {
        lab.app_mut().update();
        let world = lab.app_mut().world_mut();
        if world.resource::<Telemetry>().history.len() > ends.len() {
            let mut robots = world.query::<(&RobotBrain, &WalkTarget)>();
            let (_, target) = robots
                .iter(world)
                .find(|(brain, _)| brain.agent_index == SHOWCASE_INDEX)
                .expect("the showcased robot walks on");
            ends.push(*target);
            if ends.len() == 2 {
                break;
            }
        }
    }
    assert_eq!(ends.len(), 2, "two generations did not finish");
    assert_ne!(ends[0].point, ends[1].point, "no new targets");
}

/// Pattern-generator brains drive robots through generations.
#[test]
fn pattern_generators_drive_robots_through_generations() {
    let config = LabConfig {
        brain: BrainEncoding::Supg,
        ..small_config(41)
    };
    let history = run(config, 2);
    assert!(history.iter().all(|s| s.evaluated == 12));
    let furthest = history
        .iter()
        .map(|s| s.max_displacement)
        .fold(0.0, f32::max);
    assert!(
        furthest > 0.05,
        "no patterned robot moved 5 cm ({furthest} m)"
    );
}

/// With the tilt sense every brain reads its body's lean, and a seed
/// repeats the run.
#[test]
fn brains_can_sense_tilt() {
    let config = || LabConfig {
        tilt_sense: true,
        terrain: Terrain::Hilly,
        ..small_config(43)
    };
    let mut lab = HeadlessLab::new(config());
    let history = lab.run(2, |_| {}).expect("finishes").history.clone();
    for _ in 0..4 {
        lab.app_mut().update();
    }
    let world = lab.app_mut().world_mut();
    let mut brains = world.query::<&RobotBrain>();
    let mut sensed = 0;
    for brain in brains.iter(world) {
        let at = brain.tilt_input.expect("a tilt input");
        let tilt = &brain.sensor_inputs[at..at + 2];
        assert!(tilt.iter().all(|t| (-1.0..=1.0).contains(t)), "{tilt:?}");
        sensed += 1;
    }
    assert!(sensed > 0);
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
