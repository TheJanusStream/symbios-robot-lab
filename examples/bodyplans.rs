//! Measures what evolution can make of a body plan:
//!
//! ```text
//! cargo run --release --example bodyplans -- describe
//! cargo run --release --example bodyplans -- mutate --samples 4000
//! cargo run --release --example bodyplans -- culls --file assets/morphologies/example_walker.lsys
//! cargo run --release --example bodyplans -- halls --hall hall.json
//! ```
//!
//! The first six modes read the lab's body plans (`core::presets`), or the
//! `.lsys` files named with `--file PATH` (repeatable) instead. The first
//! five need no physics and run in a second or two.
//!
//! - `describe` prints each plan as the lab stores it and the bodies it
//!   grows: parts, driven joints, feet, mass and size for each number of
//!   derivation steps, how many different bodies its seeds give, and which
//!   of its constants change the body when scaled by a fifth (a constant
//!   only rules use changes nothing: the L-system copied its value into them
//!   when it read them).
//! - `shape` lists the parts and joints of each plan after `--steps`
//!   derivation steps [the lab's starting 5], where they sit when the robot
//!   spawns; with `--hall PATH`, of each hall's best robot instead.
//! - `mutate` mutates each plan's body `--samples` times [2000], as breeding
//!   does, and counts what came of it: nothing, the same body from a changed
//!   source, a body of the same build with other measures, a body of another
//!   build, or a body the lab cannot score; and what the bodies of another
//!   build are (their parts, driven joints, feet and knees), commonest
//!   first.
//! - `walk` chains `--steps` mutations [40], keeping each body the lab can
//!   score, `--walks` times [200]: how far unselected variation carries a
//!   plan.
//! - `cross` crosses every pair of plans `--samples` times.
//! - `culls` walks `--samples` mutants of each plan for one generation, each
//!   with a fresh brain, and prints what mutation changed in those the
//!   physics culled.
//! - `gait` walks the best `--top` robots [1] of each hall of fame saved by
//!   `evolve --save-hall` (`--hall PATH`, repeatable) once and prints the
//!   factors of their fitness.
//! - `halls` prints the bodies of each hall's best `--top` robots (their
//!   parts, driven joints, feet, legs and knees: the hinges that split a leg
//!   ending in a foot) and the plan each descends from.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;
use std::error::Error;
use symbios::System;
use symbios_genetics::scorers::Scorer;
use symbios_robot::{JointType, SensorType, ShapePrimitive};
use symbios_robot_lab::core::config::{BrainEncoding, LabConfig};
use symbios_robot_lab::core::descriptor::BodyTraits;
use symbios_robot_lab::core::genome::{
    AgentGenotype, INITIAL_LSYSTEM_ITERATIONS, MAX_LSYSTEM_ITERATIONS,
};
use symbios_robot_lab::core::presets::{MORPHOLOGY_POOL, PLAN_NAMES};
use symbios_robot_lab::headless::HeadlessLab;
use symbios_robot_lab::simulation::body::{legs, modules_in_order};
use symbios_robot_lab::simulation::clock::GenerationClock;
use symbios_robot_lab::simulation::fitness::{
    ComBounceReward, CostOfTransport, Economy, FootAirTimeFactor, JerkReward, OrientationStability,
    SteadyCommands, VelocityGate, trajectory_of, walker_fitness,
};
use symbios_robot_lab::simulation::generation::MIN_MOTORS;
use symbios_robot_lab::simulation::hall_of_fame::Champion;
use symbios_robot_lab::simulation::hyperneat::fresh_cppn;
use symbios_robot_lab::simulation::lifecycle::SpawnRequest;
use symbios_robot_lab::simulation::lifecycle::{DerivedBody, derive_blueprint};
use symbios_robot_lab::simulation::physics::FIXED_HZ;
use symbios_robot_lab::simulation::spawn::spawn_rotation;

/// The mutation rate breeding uses by default.
const RATE: f32 = 0.3;
/// Two measures closer than this are the same.
const SAME: f32 = 1e-4;
/// Mutations a `walk` chains unless `--steps` says otherwise.
const WALK_STEPS: usize = 40;
/// What to say when the mode is missing or unknown.
const MODES: &str = "give a mode: describe, shape, mutate, walk, cross, culls, gait or halls";

struct Plan {
    name: String,
    source: String,
}

/// A body reduced to what tells two bodies apart.
#[derive(Clone, PartialEq)]
struct Body {
    /// One entry per part and joint: what kind, hanging from which part.
    build: Vec<String>,
    /// Every length, angle and limit, in the order of `build`.
    measures: Vec<f32>,
    parts: usize,
    motors: usize,
    feet: usize,
    legs: usize,
    /// Joints that split a leg: every hinge after the first on a leg that
    /// ends in a foot (a part with a touch sensor).
    knees: usize,
    mass: f32,
    length: f32,
    height: f32,
}

impl Body {
    fn of(derived: &DerivedBody) -> Self {
        let blueprint = &derived.blueprint;
        let mut build = Vec::new();
        let mut measures = Vec::new();
        for (id, module) in modules_in_order(blueprint) {
            let (kind, dims): (&str, Vec<f32>) = match module.shape {
                ShapePrimitive::Box(h) => ("box", vec![h.x, h.y, h.z]),
                ShapePrimitive::Cylinder { radius, height } => ("cyl", vec![radius, height]),
                ShapePrimitive::Sphere(r) => ("orb", vec![r]),
                ShapePrimitive::Capsule { radius, height } => ("cap", vec![radius, height]),
            };
            let sensors: String = module
                .sensors
                .iter()
                .map(|s| match s.sensor_type {
                    SensorType::IMU => 'i',
                    SensorType::Touch => 't',
                    SensorType::Ultrasonic => 'u',
                    SensorType::Camera | SensorType::Lidar => '?',
                })
                .collect();
            build.push(format!("{id}:{kind}:{sensors}"));
            measures.extend(dims);
            let (position, rotation) = module.transform;
            measures.extend(position.to_array());
            measures.extend(rotation.to_array());
        }
        for joint in &blueprint.joints {
            let kind = match joint.joint_type {
                JointType::Fixed => "fixed",
                JointType::Ball => "ball",
                JointType::Hinge { .. } => "hinge",
                JointType::Prismatic { .. } => "slider",
                JointType::Screw { .. } => "screw",
            };
            build.push(format!("{}>{}:{kind}", joint.parent_id, joint.child_id));
            if let Some(axis) = joint.joint_type.axis() {
                measures.extend(axis.to_array());
            }
            if let Some(limit) = joint.limits.last() {
                measures.extend([limit.min, limit.max, limit.effort.clamp(0.0, 5000.0)]);
            }
        }
        let traits = BodyTraits::of(blueprint);
        let aabb = blueprint.aabb(spawn_rotation());
        let leg_of = legs(blueprint);
        let mut hinges_on: BTreeMap<usize, usize> = BTreeMap::new();
        for joint in &blueprint.joints {
            if let (JointType::Hinge { .. }, Some(&leg)) =
                (&joint.joint_type, leg_of.get(&joint.child_id))
            {
                *hinges_on.entry(leg).or_default() += 1;
            }
        }
        let knees = blueprint
            .modules
            .iter()
            .filter(|(_, module)| {
                let touch = |s: &symbios_robot::SensorMount| s.sensor_type == SensorType::Touch;
                module.sensors.iter().any(touch)
            })
            .filter_map(|(id, _)| leg_of.get(id))
            .map(|leg| hinges_on.get(leg).copied().unwrap_or(0).saturating_sub(1))
            .sum();
        Self {
            build,
            measures,
            parts: blueprint.modules.len(),
            motors: derived.num_outputs,
            feet: traits.feet,
            legs: leg_of.values().max().map_or(0, |&n| n + 1),
            knees,
            mass: blueprint.modules.values().map(|m| m.mass).sum(),
            length: traits.length,
            height: aabb.max.y - aabb.min.y,
        }
    }

    fn scorable(&self) -> bool {
        self.motors > MIN_MOTORS
    }

    fn same_measures(&self, other: &Self) -> bool {
        self.measures.len() == other.measures.len()
            && self
                .measures
                .iter()
                .zip(&other.measures)
                .all(|(a, b)| (a - b).abs() <= SAME * a.abs().max(1.0))
    }

    fn summary(&self) -> String {
        format!(
            "{:>3} parts {:>3} motors {:>2} feet {:>2} legs {:>2} knees {:>6.1} kg {:>4.1} m long {:>4.1} m high",
            self.parts,
            self.motors,
            self.feet,
            self.legs,
            self.knees,
            self.mass,
            self.length,
            self.height
        )
    }
}

fn body_of(genotype: &AgentGenotype) -> Option<Body> {
    derive_blueprint(
        &genotype.morphology,
        genotype.lsystem_seed,
        genotype.lsystem_iterations,
    )
    .map(|derived| Body::of(&derived))
}

fn genotype_of(plan: &Plan, seed: u64, rng: &mut Pcg64) -> AgentGenotype {
    AgentGenotype::new(
        plan.source.clone(),
        seed,
        BrainEncoding::HyperNeat,
        fresh_cppn(rng),
    )
}

/// What a variation made of a body.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Outcome {
    /// The genotype did not change.
    Untouched,
    /// The genotype changed, the body did not.
    SameBody,
    /// The same build with other measures.
    Tuned,
    /// Another build.
    Rebuilt,
    /// No body, or one too poor in motors to be scored.
    Lost,
}

fn outcome(
    parent: &AgentGenotype,
    before: &Body,
    child: &AgentGenotype,
) -> (Outcome, Option<Body>) {
    if parent.same_body(child) {
        return (Outcome::Untouched, Some(before.clone()));
    }
    let Some(after) = body_of(child).filter(Body::scorable) else {
        return (Outcome::Lost, None);
    };
    let kind = if after.build != before.build {
        Outcome::Rebuilt
    } else if after.same_measures(before) {
        Outcome::SameBody
    } else {
        Outcome::Tuned
    };
    (kind, Some(after))
}

fn percent(count: usize, of: usize) -> f32 {
    100.0 * count as f32 / of.max(1) as f32
}

fn describe(plans: &[Plan]) {
    let mut rng = Pcg64::seed_from_u64(1);
    for plan in plans {
        let genotype = genotype_of(plan, 1, &mut rng);
        println!("== {} ==", plan.name);
        println!("{}", genotype.morphology);
        for iterations in 1..=MAX_LSYSTEM_ITERATIONS {
            let mut distinct: Vec<Body> = Vec::new();
            for seed in 1..=64 {
                let body =
                    derive_blueprint(&genotype.morphology, seed, iterations).map(|d| Body::of(&d));
                if let Some(body) = body
                    && !distinct
                        .iter()
                        .any(|b| b.build == body.build && b.same_measures(&body))
                {
                    distinct.push(body);
                }
            }
            match distinct.first() {
                Some(first) => println!(
                    "  {iterations} steps: {} ({} bodies over 64 seeds, motors {}..{})",
                    first.summary(),
                    distinct.len(),
                    distinct.iter().map(|b| b.motors).min().unwrap_or(0),
                    distinct.iter().map(|b| b.motors).max().unwrap_or(0),
                ),
                None => println!("  {iterations} steps: no body"),
            }
        }
        // Which constants reach the body: each scaled from a half to double,
        // since one that sets a threshold changes nothing until it passes it.
        let Ok(system) = System::from_source(&genotype.morphology) else {
            println!("  does not parse");
            continue;
        };
        let Some(base) = body_of(&genotype) else {
            continue;
        };
        let mut names: Vec<&String> = system.constants.keys().collect();
        names.sort();
        let (mut live, mut dead) = (Vec::new(), Vec::new());
        for name in names {
            let changes = [0.5, 0.8, 1.2, 2.0].iter().any(|factor| {
                let mut varied = System::from_source(&genotype.morphology).expect("parsed above");
                if let Some(value) = varied.constants.get_mut(name) {
                    *value *= factor;
                }
                let child = AgentGenotype {
                    morphology: varied.to_source(),
                    ..genotype.clone()
                };
                body_of(&child).is_none_or(|b| b.build != base.build || !b.same_measures(&base))
            });
            if changes {
                live.push(name.as_str());
            } else {
                dead.push(name.as_str());
            }
        }
        println!("  constants that change the body: {}", live.join(" "));
        println!(
            "  constants that change nothing, halved to doubled: {}",
            dead.join(" ")
        );
    }
}

fn mutate(plans: &[Plan], samples: usize) {
    println!(
        "{:<12} {:>9} {:>9} {:>7} {:>8} {:>6}   of the rebuilt bodies: parts, motors, feet (mean change)",
        "plan", "untouched", "same body", "tuned", "rebuilt", "lost"
    );
    for plan in plans {
        let mut rng = Pcg64::seed_from_u64(7);
        let mut counts: BTreeMap<Outcome, usize> = BTreeMap::new();
        let mut builds: BTreeMap<(usize, usize, usize, usize), usize> = BTreeMap::new();
        let (mut d_parts, mut d_motors, mut d_feet, mut rebuilt) = (0.0, 0.0, 0.0, 0usize);
        for sample in 0..samples {
            let parent = genotype_of(plan, sample as u64 + 1, &mut rng);
            let Some(before) = body_of(&parent) else {
                continue;
            };
            let mut child = parent.clone();
            child.mutate_body(&mut rng, RATE);
            let (kind, after) = outcome(&parent, &before, &child);
            *counts.entry(kind).or_default() += 1;
            if let (Outcome::Rebuilt, Some(after)) = (kind, after) {
                rebuilt += 1;
                d_parts += after.parts as f32 - before.parts as f32;
                d_motors += after.motors as f32 - before.motors as f32;
                d_feet += after.feet as f32 - before.feet as f32;
                *builds
                    .entry((after.parts, after.motors, after.feet, after.knees))
                    .or_default() += 1;
            }
        }
        let total: usize = counts.values().sum();
        let share = |kind| percent(counts.get(&kind).copied().unwrap_or(0), total);
        let mean = |sum: f32| sum / rebuilt.max(1) as f32;
        println!(
            "{:<12} {:>8.1}% {:>8.1}% {:>6.1}% {:>7.1}% {:>5.1}%   {:+.2} {:+.2} {:+.2}",
            plan.name,
            share(Outcome::Untouched),
            share(Outcome::SameBody),
            share(Outcome::Tuned),
            share(Outcome::Rebuilt),
            share(Outcome::Lost),
            mean(d_parts),
            mean(d_motors),
            mean(d_feet),
        );
        // What the rebuilt bodies are: the commonest builds, by their
        // parts, driven joints, feet and knees.
        let mut commonest: Vec<_> = builds.into_iter().collect();
        commonest.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        let listed: Vec<String> = commonest
            .iter()
            .take(6)
            .map(|&((parts, motors, feet, knees), count)| {
                format!(
                    "{parts} parts {motors} motors {feet} feet {knees} knees {:.1}%",
                    percent(count, total)
                )
            })
            .collect();
        if !listed.is_empty() {
            println!("{:<12} rebuilt into: {}", "", listed.join(" | "));
        }
    }
}

fn walk(plans: &[Plan], steps: usize, walks: usize) {
    println!(
        "{:<12} {:>6} {:>8}   after {steps} mutations: builds reached, parts, motors, feet, legs (min mean max)",
        "plan", "lost", "rebuilt"
    );
    for plan in plans {
        let mut rng = Pcg64::seed_from_u64(11);
        let (mut lost, mut rebuilt, mut tried) = (0usize, 0usize, 0usize);
        let mut ends: Vec<Body> = Vec::new();
        let mut builds: Vec<Vec<String>> = Vec::new();
        for walk in 0..walks {
            let mut current = genotype_of(plan, walk as u64 + 1, &mut rng);
            let Some(mut body) = body_of(&current) else {
                continue;
            };
            for _ in 0..steps {
                let mut child = current.clone();
                child.mutate_body(&mut rng, RATE);
                tried += 1;
                match outcome(&current, &body, &child) {
                    (Outcome::Lost, _) => lost += 1,
                    (kind, Some(after)) => {
                        rebuilt += usize::from(kind == Outcome::Rebuilt);
                        current = child;
                        body = after;
                    }
                    (_, None) => {}
                }
            }
            if !builds.contains(&body.build) {
                builds.push(body.build.clone());
            }
            ends.push(body);
        }
        let range = |f: &dyn Fn(&Body) -> f32| {
            let values: Vec<f32> = ends.iter().map(f).collect();
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
            format!("{min:.0} {mean:.1} {max:.0}")
        };
        println!(
            "{:<12} {:>5.1}% {:>7.1}%   {:>4} | {} | {} | {} | {}",
            plan.name,
            percent(lost, tried),
            percent(rebuilt, tried),
            builds.len(),
            range(&|b| b.parts as f32),
            range(&|b| b.motors as f32),
            range(&|b| b.feet as f32),
            range(&|b| b.legs as f32),
        );
    }
}

fn cross(plans: &[Plan], samples: usize) {
    println!("rows: the parent whose axiom the child keeps; columns: the other parent");
    println!("each cell: % lost / % a body unlike both parents");
    print!("{:<12}", "");
    for plan in plans {
        print!(" {:>11}", plan.name.chars().take(11).collect::<String>());
    }
    println!();
    let mut rng = Pcg64::seed_from_u64(13);
    for a in plans {
        print!("{:<12}", a.name);
        for b in plans {
            let (mut lost, mut novel, mut total) = (0usize, 0usize, 0usize);
            for sample in 0..samples {
                let pa = genotype_of(a, sample as u64 + 1, &mut rng);
                let pb = genotype_of(b, sample as u64 + 101, &mut rng);
                let (Some(body_a), Some(body_b)) = (body_of(&pa), body_of(&pb)) else {
                    continue;
                };
                // Rate 0: the cross alone, with no mutation after it.
                let child = pa.body_child(&pb, 0.0, &mut rng);
                total += 1;
                match body_of(&child).filter(Body::scorable) {
                    None => lost += 1,
                    Some(body) => {
                        let unlike = |p: &Body| body.build != p.build || !body.same_measures(p);
                        novel += usize::from(unlike(&body_a) && unlike(&body_b));
                    }
                }
            }
            print!(
                " {:>5.0}/{:<5.0}",
                percent(lost, total),
                percent(novel, total)
            );
        }
        println!();
    }
}

/// Prints where each plan's parts and joints sit when it spawns, after
/// `steps` derivation steps; with halls, each hall's best robot as it was
/// bred instead.
fn shape(plans: &[Plan], steps: usize, halls: &[String]) -> Result<(), Box<dyn Error>> {
    if halls.is_empty() {
        let mut rng = Pcg64::seed_from_u64(1);
        for plan in plans {
            let source = genotype_of(plan, 1, &mut rng).morphology;
            print_shape(
                &format!("{} at {steps} steps", plan.name),
                &source,
                1,
                steps,
            );
        }
        return Ok(());
    }
    for path in halls {
        let champions = Champion::all_from_json(&std::fs::read_to_string(path)?)
            .map_err(|err| format!("{path}: {err}"))?;
        if let Some(best) = champions.first() {
            let g = &best.genotype;
            print_shape(path, &g.morphology, g.lsystem_seed, g.lsystem_iterations);
        }
    }
    Ok(())
}

/// One body's parts and joints: x forward, y up, z to the side, the lowest
/// point of the body at y = 0.
fn print_shape(name: &str, source: &str, seed: u64, steps: usize) {
    println!("== {name} ==");
    let Some(derived) = derive_blueprint(source, seed, steps) else {
        println!("  no body");
        return;
    };
    let blueprint = &derived.blueprint;
    let turn = spawn_rotation();
    let floor = blueprint.aabb(turn).min.y;
    let at = |p: bevy::math::Vec3| {
        let w = turn * p;
        format!("({:+.2} {:+.2} {:+.2})", w.x, w.y - floor, w.z)
    };
    let dir = |d: bevy::math::Vec3| {
        let w = turn * d;
        format!("({:+.2} {:+.2} {:+.2})", w.x, w.y, w.z)
    };
    for (id, module) in modules_in_order(blueprint) {
        let (position, rotation) = module.transform;
        let sensors: Vec<String> = module
            .sensors
            .iter()
            .map(|s| format!("{:?}", s.sensor_type))
            .collect();
        println!(
            "  part {id:>2} {:<44} centre {} axis {} {:.2} kg {}",
            format!("{:?}", module.shape),
            at(position),
            dir(rotation * bevy::math::Vec3::Y),
            module.mass,
            sensors.join(" ")
        );
    }
    for joint in &blueprint.joints {
        let parent = blueprint.modules.get(&joint.parent_id);
        let anchor = parent.map_or_else(String::new, |m| {
            at(m.transform.0 + m.transform.1 * joint.anchor_parent)
        });
        let axis = joint
            .joint_type
            .axis()
            .zip(parent)
            .map_or_else(String::new, |(axis, m)| dir(m.transform.1 * axis));
        let limit = joint.limits.last().map_or_else(String::new, |l| {
            format!("{:+.2}..{:+.2} rad or m, {:.0}", l.min, l.max, l.effort)
        });
        println!(
            "  joint {:>2}>{:<2} {:<7} at {anchor} axis {axis} {limit}",
            joint.parent_id,
            joint.child_id,
            format!("{:?}", joint.joint_type.kind()),
        );
    }
}

/// Walks `robots` for one generation, each as bred, in a lab set up from
/// `config`, and returns the lab with their scores, or `None` if the
/// generation did not end (the lab's clock or step size changed under this
/// example).
fn walk_one_generation(config: LabConfig, robots: Vec<AgentGenotype>) -> Option<HeadlessLab> {
    let steps = (f64::from(config.generation_duration) * FIXED_HZ).ceil() as usize;
    let mut lab = HeadlessLab::new(config);
    lab.app_mut()
        .world_mut()
        .resource_mut::<LabConfig>()
        .auto_restart = false;
    for robot in robots {
        lab.send_into_evolution(robot);
    }
    lab.app_mut().world_mut().write_message(SpawnRequest);
    // An update advances at least one fixed step, so `steps` updates and a
    // few to spawn are enough whatever the lab's steps per update.
    for _ in 0..steps + 16 {
        lab.app_mut().update();
        let clock = lab.app_mut().world().resource::<GenerationClock>();
        if clock.is_finished() {
            return Some(lab);
        }
    }
    None
}

/// Walks each hall's best `top` robots once, two clones each, and prints
/// what their fitness is made of.
fn gait(paths: &[String], top: usize) -> Result<(), Box<dyn Error>> {
    println!(
        "{:<28} {:>7} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>5} {:>7} {:>6}",
        "robot",
        "fitness",
        "metres",
        "m/s",
        "orient",
        "steady",
        "bjerk",
        "vgate",
        "feet",
        "bounce",
        "kg",
        "duty",
        "economy",
        "CoT"
    );
    for path in paths {
        let champions = Champion::all_from_json(&std::fs::read_to_string(path)?)
            .map_err(|err| format!("{path}: {err}"))?;
        for (rank, champion) in champions.iter().take(top).enumerate() {
            let config = LabConfig {
                population_size: 2,
                seed: Some(0xF2E5_0000_0000),
                ..LabConfig::default()
            };
            let clones = vec![champion.genotype.clone(); 2];
            let lab = walk_one_generation(config, clones)
                .ok_or("a generation did not end: the walk cannot be scored")?;
            let name = std::path::Path::new(path)
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            for agent in &lab.population().agents {
                let t = trajectory_of(agent);
                println!(
                    "{:<28} {:>7.3} {:>6.2} {:>6.2} | {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} | {:>6.1} {:>5.2} {:>7.3} {:>6.2}{}",
                    format!("{name}#{rank}"),
                    walker_fitness().score(&t),
                    t.forward_displacement,
                    t.avg_velocity,
                    OrientationStability.score(&t),
                    SteadyCommands.score(&t),
                    JerkReward.score(&t),
                    VelocityGate.score(&t),
                    FootAirTimeFactor.score(&t),
                    ComBounceReward.score(&t),
                    t.mass,
                    t.support_duty_factor,
                    Economy.score(&t),
                    CostOfTransport.score(&t),
                    if agent.exploded { " EXPLODED" } else { "" },
                );
            }
        }
    }
    Ok(())
}

/// Walks `samples` mutants of each plan for one generation, each with a
/// fresh brain, prints what mutation changed in those the physics culled,
/// and sets each change's count among the culled against its count among
/// the mutants that walked on: a change is suspect only if it is commoner
/// among the culled.
fn culls(plans: &[Plan], samples: usize) -> Result<(), Box<dyn Error>> {
    const BATCH: usize = 64;
    for plan in plans {
        let mut rng = Pcg64::seed_from_u64(17);
        let original = genotype_of(plan, 1, &mut rng);
        let (mut walked, mut culled, mut unscored) = (0usize, 0usize, 0usize);
        // Per change: how many culled mutants carry it, how many others.
        let mut changes: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        while walked < samples {
            let config = LabConfig {
                population_size: BATCH,
                seed: Some(0xC0FFEE + walked as u64),
                ..LabConfig::default()
            };
            let mutants = (0..BATCH)
                .map(|_| {
                    let mut mutant = genotype_of(plan, 1, &mut rng);
                    mutant.mutate_body(&mut rng, RATE);
                    mutant
                })
                .collect();
            let lab = walk_one_generation(config, mutants)
                .ok_or("a generation did not end: the mutants cannot be judged")?;
            for agent in &lab.population().agents {
                walked += 1;
                unscored += usize::from(agent.num_motors <= MIN_MOTORS);
                culled += usize::from(agent.exploded);
                let changed = changed_lines(&original.morphology, &agent.genotype.morphology);
                if agent.exploded {
                    println!(
                        "culled ({} motors, {:.0} kg, steps {}):",
                        agent.num_motors, agent.mass, agent.genotype.lsystem_iterations
                    );
                    for line in &changed {
                        println!("    {line}");
                    }
                }
                let mut keys: Vec<String> = changed.iter().map(|line| change_of(line)).collect();
                keys.sort();
                keys.dedup();
                for key in keys {
                    let count = changes.entry(key).or_default();
                    if agent.exploded {
                        count.0 += 1;
                    } else {
                        count.1 += 1;
                    }
                }
            }
        }
        let survived = walked - culled;
        println!(
            "{}: {culled} of {walked} mutants culled ({:.1}%), {unscored} too poor in motors",
            plan.name,
            percent(culled, walked)
        );
        let mut changes: Vec<(String, (usize, usize))> = changes.into_iter().collect();
        changes.sort_by_key(|(_, (among_culled, _))| std::cmp::Reverse(*among_culled));
        for (key, (among_culled, among_others)) in changes.iter().take(12) {
            println!(
                "  {key}: changed in {:.0}% of the culled, {:.0}% of the others",
                percent(*among_culled, culled),
                percent(*among_others, survived)
            );
        }
    }
    Ok(())
}

/// What a changed source line changes: the constant it defines, or the
/// rule it belongs to (its predecessor and condition, without the
/// probability mutation may have written before them).
fn change_of(line: &str) -> String {
    if let Some(rest) = line.strip_prefix("#define ") {
        let name = rest.split_whitespace().next().unwrap_or(rest);
        return format!("constant {name}");
    }
    let head = line.split("->").next().unwrap_or(line).trim();
    // "0.93 : L(n, m) : n > 50" and "L(n, m) : n > 50" are the same rule.
    let rule = match head.split_once(" : ") {
        Some((before, after)) if before.parse::<f64>().is_ok() => after,
        _ => head,
    };
    format!("rule {rule}")
}

/// The lines of `after` that `before` lacks.
fn changed_lines(before: &str, after: &str) -> Vec<String> {
    let known: Vec<&str> = before.lines().collect();
    after
        .lines()
        .filter(|line| !known.contains(line))
        .map(str::to_string)
        .collect()
}

/// The `omega:` line of a source: breeding never changes it except by the
/// constants it names, so it tells which plan a robot descends from.
fn axiom_of(source: &str) -> &str {
    source
        .lines()
        .find(|line| line.trim_start().starts_with("omega"))
        .unwrap_or("")
        .trim()
}

fn halls(plans: &[Plan], paths: &[String], top: usize) -> Result<(), Box<dyn Error>> {
    let mut rng = Pcg64::seed_from_u64(1);
    // Plans whose axioms read alike (the same layout with other constants or
    // another tail) cannot be told apart in a descendant: they share a label.
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    for plan in plans {
        let axiom = axiom_of(&genotype_of(plan, 1, &mut rng).morphology).to_string();
        let label = labels.entry(axiom).or_default();
        if !label.is_empty() {
            label.push_str(" or ");
        }
        label.push_str(&plan.name);
    }
    let mut founders: BTreeMap<&str, usize> = BTreeMap::new();
    let mut sums = [0.0f32; 8];
    let mut counted = 0usize;
    println!(
        "hall,rank,fitness,generation,plan,steps,parts,motors,feet,legs,knees,mass,length,height"
    );
    for path in paths {
        let champions = Champion::all_from_json(&std::fs::read_to_string(path)?)
            .map_err(|err| format!("{path}: {err}"))?;
        for (rank, champion) in champions.iter().take(top).enumerate() {
            let genotype = &champion.genotype;
            let Some(body) = body_of(genotype) else {
                continue;
            };
            let name = labels
                .get(axiom_of(&genotype.morphology))
                .map_or("another plan", String::as_str);
            *founders.entry(name).or_default() += 1;
            println!(
                "{path},{rank},{:.3},{},{name},{},{},{},{},{},{},{:.1},{:.2},{:.2}",
                champion.fitness.unwrap_or(0.0),
                champion.generation.unwrap_or(0),
                genotype.lsystem_iterations,
                body.parts,
                body.motors,
                body.feet,
                body.legs,
                body.knees,
                body.mass,
                body.length,
                body.height,
            );
            let values = [
                champion.fitness.unwrap_or(0.0),
                genotype.lsystem_iterations as f32,
                body.parts as f32,
                body.motors as f32,
                body.feet as f32,
                body.mass,
                body.length,
                body.height,
            ];
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value;
            }
            counted += 1;
        }
    }
    let n = counted.max(1) as f32;
    println!(
        "mean of {counted}: fitness {:.3}, steps {:.2}, parts {:.1}, motors {:.1}, feet {:.1}, mass {:.1} kg, length {:.2} m, height {:.2} m",
        sums[0] / n,
        sums[1] / n,
        sums[2] / n,
        sums[3] / n,
        sums[4] / n,
        sums[5] / n,
        sums[6] / n,
        sums[7] / n,
    );
    for (name, count) in &founders {
        println!("descend from {name}: {count}");
    }
    Ok(())
}

fn plans_from(files: &[String]) -> Result<Vec<Plan>, Box<dyn Error>> {
    if files.is_empty() {
        return Ok(PLAN_NAMES
            .iter()
            .zip(MORPHOLOGY_POOL)
            .map(|(name, source)| Plan {
                name: (*name).to_string(),
                source: (*source).to_string(),
            })
            .collect());
    }
    files
        .iter()
        .map(|path| {
            let source = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let name = std::path::Path::new(path)
                .file_stem()
                .map_or_else(|| path.clone(), |s| s.to_string_lossy().into_owned());
            Ok(Plan { name, source })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or(MODES)?;
    let (mut files, mut samples, mut walks) = (Vec::new(), 2000usize, 200usize);
    let (mut hall_paths, mut top, mut steps) = (Vec::new(), 1usize, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--file" => files.push(value()?),
            "--samples" => samples = value()?.parse()?,
            "--steps" => steps = Some(value()?.parse()?),
            "--walks" => walks = value()?.parse()?,
            "--hall" => hall_paths.push(value()?),
            "--top" => top = value()?.parse()?,
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let plans = plans_from(&files)?;
    match mode.as_str() {
        "describe" => describe(&plans),
        "mutate" => mutate(&plans, samples),
        "walk" => walk(&plans, steps.unwrap_or(WALK_STEPS), walks),
        "cross" => cross(&plans, samples),
        "halls" => halls(&plans, &hall_paths, top)?,
        "shape" => shape(
            &plans,
            steps.unwrap_or(INITIAL_LSYSTEM_ITERATIONS),
            &hall_paths,
        )?,
        "gait" => gait(&hall_paths, top)?,
        "culls" => culls(&plans, samples)?,
        other => return Err(format!("unknown mode {other}; {MODES}").into()),
    }
    Ok(())
}
