//! Integration tests for the L-System morphology → RobotBlueprint pipeline.
//!
//! These tests verify that L-System source strings produce valid robot
//! topologies with the expected number of modules, joints, and joint types.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios::System;
use symbios::vm::Op;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter, SensorType};
use symbios_robot_lab::core::config::BrainEncoding;
use symbios_robot_lab::core::descriptor::BodyTraits;
use symbios_robot_lab::core::genome::{AgentGenotype, INITIAL_LSYSTEM_ITERATIONS};
use symbios_robot_lab::core::presets::{MORPHOLOGY_POOL, PLAN_NAMES};
use symbios_robot_lab::simulation::body::legs;
use symbios_robot_lab::simulation::generation::MIN_MOTORS;
use symbios_robot_lab::simulation::hyperneat::fresh_cppn;
use symbios_robot_lab::simulation::lifecycle::{DerivedBody, derive_blueprint};

/// The bodies the unit tests use too (`core::fixtures` is compiled for tests
/// only, so this crate includes the file itself).
#[path = "../src/core/fixtures.rs"]
mod fixtures;

/// Parse an L-System morphology string and build a RobotBlueprint.
fn build_blueprint_from_source(source: &str) -> symbios_robot::RobotBlueprint {
    let mut sys = System::from_source(source).expect("valid L-System source");
    sys.derive(3).expect("derivation");

    let mut interpreter = RobotInterpreter::new(RobotConfig::default());
    interpreter.populate_standard_symbols(&sys.interner);
    interpreter.build_blueprint(&sys.state)
}

/// Count hinge joints in a blueprint (used for NEAT output sizing).
fn count_hinge_joints(bp: &symbios_robot::RobotBlueprint) -> usize {
    bp.joints
        .iter()
        .filter(|j| matches!(j.joint_type, JointType::Hinge { .. }))
        .count()
}

// ---------------------------------------------------------------------------
// Simple morphology tests
// ---------------------------------------------------------------------------

#[test]
fn simple_snake_produces_correct_topology() {
    let bp = build_blueprint_from_source("omega: B(1) J B(1) J B(1)");
    assert_eq!(bp.modules.len(), 3);
    assert_eq!(bp.joints.len(), 2);
    assert_eq!(count_hinge_joints(&bp), 2);
}

#[test]
fn single_module_no_joints() {
    let bp = build_blueprint_from_source("omega: B(1)");
    assert_eq!(bp.modules.len(), 1);
    assert_eq!(bp.joints.len(), 0);
}

#[test]
fn branching_creates_independent_limbs() {
    // Two branches from a central body
    let bp = build_blueprint_from_source("omega: B(1) [ J B(1) ] [ +(90) J B(1) ]");
    assert_eq!(bp.modules.len(), 3, "1 body + 2 branch segments");
    assert_eq!(bp.joints.len(), 2, "One joint per branch");
}

// ---------------------------------------------------------------------------
// Rule-based morphology tests
// ---------------------------------------------------------------------------

#[test]
fn rule_based_leg_expands_correctly() {
    let source = "\
#define len 0.5
omega: B(1) [ L ]
L -> J B(len) J B(len)";
    let bp = build_blueprint_from_source(source);
    assert_eq!(bp.modules.len(), 3, "1 body + 2 leg segments from L rule");
    assert_eq!(bp.joints.len(), 2, "2 joints from L rule expansion");
}

// ---------------------------------------------------------------------------
// Joints and sensors the body plans do not all use
// ---------------------------------------------------------------------------

#[test]
fn prismatic_joint_symbol_produces_prismatic_joint() {
    // Jp sets the next J's type to Prismatic; the resulting blueprint joint
    // should match JointType::Prismatic { .. }.
    let bp = build_blueprint_from_source("omega: B(1) Ja(0,1,0) Jl(-0.5,0.5,1000,2) Jp K(0.5)");
    assert_eq!(bp.modules.len(), 2);
    assert_eq!(bp.joints.len(), 1);
    assert!(
        matches!(bp.joints[0].joint_type, JointType::Prismatic { .. }),
        "Expected Prismatic joint, got {:?}",
        bp.joints[0].joint_type
    );
}

#[test]
fn end_effector_symbol_pins_ee_to_module() {
    // E(id) marks the end-effector at the turtle's current pose, pinned to
    // the module the turtle is standing on. One module + one EE.
    let bp = build_blueprint_from_source("omega: B(1) E(7)");
    assert_eq!(bp.modules.len(), 1);
    assert_eq!(bp.end_effectors.len(), 1);
    assert_eq!(bp.end_effectors[0].id, 7);
}

#[test]
fn ultrasonic_sensor_mounts_on_current_module() {
    // Su mounts an Ultrasonic sensor on the turtle's current module.
    let bp = build_blueprint_from_source("omega: B(1) Su");
    assert_eq!(bp.modules.len(), 1);
    assert_eq!(count_sensors(&bp, SensorType::Ultrasonic), 1);
}

/// Which way each sensor of `kind` on the body looks once the robot is
/// turned over to spawn.
fn gazes(bp: &symbios_robot::RobotBlueprint, kind: SensorType) -> Vec<bevy::math::Vec3> {
    let spawned = bevy::math::Quat::from_rotation_x(-std::f32::consts::PI);
    bp.modules
        .values()
        .flat_map(|m| m.sensors.iter().map(move |s| (m, s)))
        .filter(|(_, s)| s.sensor_type == kind)
        .map(|(m, s)| spawned * m.transform.1 * s.local_rotation * bevy::math::Vec3::Y)
        .collect()
}

/// The range each sliding joint moves within: the last limit about its own
/// axis.
fn slider_ranges(bp: &symbios_robot::RobotBlueprint) -> Vec<(f32, f32)> {
    bp.joints
        .iter()
        .filter_map(|joint| match joint.joint_type {
            JointType::Prismatic { axis } => joint
                .limits
                .iter()
                .rev()
                .find(|l| l.axis.normalize().dot(axis.normalize()).abs() > 0.999)
                .map(|l| (l.min, l.max)),
            _ => None,
        })
        .collect()
}

fn count_sensors(bp: &symbios_robot::RobotBlueprint, kind: SensorType) -> usize {
    bp.modules
        .values()
        .flat_map(|m| m.sensors.iter())
        .filter(|s| s.sensor_type == kind)
        .count()
}

/// Legs that end in a touch sensor and are moved by more than one joint.
fn kneed_legs(bp: &symbios_robot::RobotBlueprint) -> usize {
    let leg_of = legs(bp);
    let mut joints_on_leg = std::collections::HashMap::new();
    for joint in &bp.joints {
        if let Some(&leg) = leg_of.get(&joint.child_id) {
            *joints_on_leg.entry(leg).or_insert(0usize) += 1;
        }
    }
    bp.modules
        .iter()
        .filter(|(_, m)| m.sensors.iter().any(|s| s.sensor_type == SensorType::Touch))
        .filter_map(|(id, _)| leg_of.get(id))
        .filter(|leg| joints_on_leg.get(leg).copied().unwrap_or(0) > 1)
        .count()
}

/// Every fixture still builds a body with driven joints: they are what the
/// tests exercise sliders, ball joints, knees and stochastic rules with.
#[test]
fn every_fixture_grows_a_driven_body() {
    for (i, source) in fixtures::ALL.iter().enumerate() {
        assert!(
            count_hinge_joints(&build_blueprint_from_source(source)) > 0,
            "fixture {i}"
        );
    }
}

#[test]
fn bodies_slide_stand_on_limited_ankles_and_see() {
    // Tripod: each of the three shins slides along its leg, within the
    // limit given for it (the last limit about the leg's axis).
    let tripod = build_blueprint_from_source(fixtures::TRIPOD);
    assert_eq!(slider_ranges(&tripod), vec![(-0.15, 0.15); 3]);

    // Strider: four ball-jointed ankles, each with exactly its own two
    // limits (the leg's hinge limits cleared).
    let strider = build_blueprint_from_source(fixtures::STRIDER);
    let ankles: Vec<_> = strider
        .joints
        .iter()
        .filter(|j| j.joint_type == JointType::Ball)
        .collect();
    assert_eq!(ankles.len(), 4);
    for ankle in ankles {
        let ranges: Vec<(f32, f32)> = ankle.limits.iter().map(|l| (l.min, l.max)).collect();
        assert_eq!(ranges, vec![(-0.3, 0.3), (-0.45, 0.45)]);
    }

    // The former scout: one ultrasonic eye, looking forward (+X, the
    // direction the walker fitness rewards) and down once the robot is
    // turned over to spawn.
    let eyes = gazes(
        &build_blueprint_from_source(fixtures::SCOUT),
        SensorType::Ultrasonic,
    );
    assert_eq!(eyes.len(), 1);
    assert!(
        eyes[0].x > 0.5 && eyes[0].y < -0.5,
        "the eye looks along {}",
        eyes[0]
    );

    // The crawler's four legs have knees, the hexapod's six too: what
    // `kneed_legs` must see for the plans' test below to mean anything.
    assert_eq!(
        kneed_legs(&build_blueprint_from_source(fixtures::CRAWLER)),
        4
    );
    assert_eq!(
        kneed_legs(&build_blueprint_from_source(fixtures::HEXAPOD)),
        6
    );
}

// ---------------------------------------------------------------------------
// The body plans: what makes a plan one evolution can build on
// (see `core::presets`)
// ---------------------------------------------------------------------------

/// A plan's name and source.
fn plans() -> impl Iterator<Item = (&'static str, &'static str)> {
    PLAN_NAMES
        .iter()
        .copied()
        .zip(MORPHOLOGY_POOL.iter().copied())
}

/// A plan's body after `steps` derivation steps.
fn body(source: &str, steps: usize) -> DerivedBody {
    derive_blueprint(source, 1, steps).expect("the plan derives")
}

/// `source` with constant `name` multiplied by `factor`, as a constant
/// mutation would leave it.
fn with_constant(source: &str, name: &str, factor: f64) -> String {
    let mut system = System::from_source(source).expect("the plan parses");
    let value = system
        .constants
        .get_mut(name)
        .unwrap_or_else(|| panic!("no constant {name}"));
    *value *= factor;
    system.to_source()
}

/// A plan's genotype as breeding handles it.
fn genotype(source: &str, rng: &mut Pcg64) -> AgentGenotype {
    AgentGenotype::new(
        source.to_string(),
        1,
        BrainEncoding::HyperNeat,
        fresh_cppn(rng),
    )
}

/// The body a genotype grows, if the lab can score it.
fn scorable(genotype: &AgentGenotype) -> Option<DerivedBody> {
    derive_blueprint(
        &genotype.morphology,
        genotype.lsystem_seed,
        genotype.lsystem_iterations,
    )
    .filter(|body| body.num_outputs > MIN_MOTORS)
}

fn feet(derived: &DerivedBody) -> usize {
    count_sensors(&derived.blueprint, SensorType::Touch)
}

/// What a body is built of: its parts, hinges, sliders and feet. Two bodies
/// of one build differ only in their measures.
fn build_of(derived: &DerivedBody) -> [usize; 4] {
    let bp = &derived.blueprint;
    let sliders = bp
        .joints
        .iter()
        .filter(|j| matches!(j.joint_type, JointType::Prismatic { .. }))
        .count();
    [
        bp.modules.len(),
        count_hinge_joints(bp),
        sliders,
        feet(derived),
    ]
}

#[test]
fn every_plan_has_a_name() {
    assert_eq!(PLAN_NAMES.len(), MORPHOLOGY_POOL.len());
}

/// In the steps a fresh body gets, every plan grows a body with motors
/// enough to be scored, an IMU on its first part (the one that carries the
/// brain) and four feet, each a stiff leg with a touch sensor: the foot
/// terms of the fitness score steps only where there are touch sensors.
#[test]
fn every_plan_grows_a_body_the_lab_scores() {
    for (name, source) in plans() {
        let derived = body(source, INITIAL_LSYSTEM_ITERATIONS);
        let bp = &derived.blueprint;
        assert!(
            derived.num_outputs > MIN_MOTORS,
            "{name}: {} motors",
            derived.num_outputs
        );
        let root = bp
            .root_module
            .and_then(|id| bp.modules.get(&id))
            .expect("a root");
        assert!(
            root.sensors
                .iter()
                .any(|s| s.sensor_type == SensorType::IMU),
            "{name}: no IMU on its first part"
        );
        assert_eq!(feet(&derived), 4, "{name}");
        assert_eq!(kneed_legs(bp), 0, "{name}: a leg has a knee");
    }
}

/// The body traits quality diversity can spread robots over: every plan
/// spans 1.5 to 6 m. Its size, the edge of a cube as roomy as the box around
/// it, is shorter than its length (a plan is longer than it is wide or
/// tall) and about 2 m.
#[test]
fn every_plan_has_a_length_on_the_scale() {
    for (name, source) in plans() {
        let traits = BodyTraits::of(&body(source, INITIAL_LSYSTEM_ITERATIONS).blueprint);
        assert_eq!(traits.feet, 4, "{name}: {traits:?}");
        assert!((1.5..6.0).contains(&traits.length), "{name}: {traits:?}");
        assert!((1.5..2.5).contains(&traits.size), "{name}: {traits:?}");
        assert!(traits.size < traits.length, "{name}: {traits:?}");
    }
}

/// Mutating a constant reaches the body only through the axiom (the rules
/// hold its value as it was when they were read), so the axiom names every
/// one of a plan's constants.
#[test]
fn a_plans_axiom_names_every_constant() {
    for (name, source) in plans() {
        let system = System::from_source(source).expect("the plan parses");
        let axiom = system.axiom_source.as_deref().expect("an axiom");
        let words: Vec<&str> = axiom
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .collect();
        for constant in system.constants.keys() {
            assert!(
                words.contains(&constant.as_str()),
                "{name}: the axiom does not use {constant}"
            );
        }
    }
}

/// Mutation flips `*` to `/` and `+` to `-` wherever a rule has one, so the
/// plans work their ratios out in the axiom. The one operator a rule keeps
/// is the trunk's count-down, whose flip grows a trunk segment per step.
#[test]
fn a_plans_rules_count_down_and_compute_nothing_else() {
    for (name, source) in plans() {
        let system = System::from_source(source).expect("the plan parses");
        let mut sums = 0;
        for rule in system.rules.values().flatten() {
            let params = rule
                .successors
                .iter()
                .flat_map(|m| m.params.iter().flatten());
            let ops: Vec<&Op> = rule.condition.iter().flatten().chain(params).collect();
            assert!(
                !ops.iter()
                    .any(|op| matches!(op, Op::Mul | Op::Div | Op::Pow)),
                "{name}: a rule multiplies or divides"
            );
            sums += ops
                .iter()
                .filter(|op| matches!(op, Op::Add | Op::Sub))
                .count();
        }
        assert_eq!(sums, 1, "{name}: its rules add or subtract {sums} times");
    }
}

/// Mutation swaps a turn for its opposite (`+` and `-`, `&` and `^`, `\`
/// and `/`, `F` and `f`) only when the plan uses both, so no plan does: a
/// flipped `-(90)` would grow every leg upward.
#[test]
fn no_plan_uses_a_turn_and_its_opposite() {
    for (name, source) in plans() {
        let system = System::from_source(source).expect("the plan parses");
        for (a, b) in [("+", "-"), ("&", "^"), ("\\", "/"), ("F", "f")] {
            let both =
                system.interner.resolve_id(a).is_some() && system.interner.resolve_id(b).is_some();
            assert!(!both, "{name} uses both {a} and {b}");
        }
    }
}

/// Both legs of a pair come from one rule, so every part off the body's
/// midline starts with its mirror image on the other side.
#[test]
fn every_plan_is_the_same_on_both_sides() {
    for (name, source) in plans() {
        let derived = body(source, INITIAL_LSYSTEM_ITERATIONS);
        let parts: Vec<(bevy::math::Vec3, f32)> = derived
            .blueprint
            .modules
            .values()
            .map(|m| (m.transform.0, m.mass))
            .collect();
        for &(at, mass) in parts.iter().filter(|(at, _)| at.z.abs() > 1e-3) {
            let mirrored = bevy::math::Vec3::new(at.x, at.y, -at.z);
            assert!(
                parts
                    .iter()
                    .any(|&(other, m)| other.distance(mirrored) < 1e-3 && (m - mass).abs() < 1e-3),
                "{name}: the part at {at} has no mirror image"
            );
        }
    }
}

/// Growth is a threshold on a constant: a tenth more trunk (half of one
/// constant mutation's reach) gives the skink another trunk segment, and a
/// fourth step that segment's legs.
#[test]
fn a_longer_trunk_grows_a_segment_and_a_step_more_its_legs() {
    let skink = MORPHOLOGY_POOL[0];
    let before = body(skink, INITIAL_LSYSTEM_ITERATIONS);

    let longer = with_constant(skink, "trunk", 1.1);
    let grown = body(&longer, INITIAL_LSYSTEM_ITERATIONS);
    assert_eq!(
        grown.num_outputs,
        before.num_outputs + 1,
        "one more trunk hinge"
    );
    assert_eq!(
        feet(&grown),
        feet(&before),
        "its legs wait for a fourth step"
    );
    let legged = body(&longer, INITIAL_LSYSTEM_ITERATIONS + 1);
    assert_eq!(feet(&legged), feet(&before) + 2);
    assert_eq!(legged.num_outputs, before.num_outputs + 3);

    // A tenth less changes nothing: the plan does not sit on a threshold.
    let shorter = body(
        &with_constant(skink, "trunk", 0.9),
        INITIAL_LSYSTEM_ITERATIONS,
    );
    assert_eq!(shorter.num_outputs, before.num_outputs);
}

/// The eel's third leg pair is laid down in three steps and grows with a
/// fourth.
#[test]
fn a_fourth_step_grows_the_eels_third_leg_pair() {
    let eel = MORPHOLOGY_POOL[2];
    let before = body(eel, INITIAL_LSYSTEM_ITERATIONS);
    let later = body(eel, INITIAL_LSYSTEM_ITERATIONS + 1);
    assert_eq!(feet(&later), feet(&before) + 2);
    assert_eq!(later.num_outputs, before.num_outputs + 2);
}

/// The scout brings the lab's ultrasonic eye into the gene pool, on a stalk
/// that reaches out past the head and looks forward and down; the eel brings
/// its slider, along the trunk and within the range given for it. Apart
/// from the eye the scout is the skink.
#[test]
fn the_scout_sees_and_the_eel_slides() {
    let skink = body(MORPHOLOGY_POOL[0], INITIAL_LSYSTEM_ITERATIONS);
    let scout = body(MORPHOLOGY_POOL[1], INITIAL_LSYSTEM_ITERATIONS);
    let eyes = gazes(&scout.blueprint, SensorType::Ultrasonic);
    assert_eq!(eyes.len(), 1);
    assert!(
        eyes[0].x > 0.5 && eyes[0].y < -0.5,
        "the eye looks along {}",
        eyes[0]
    );
    assert_eq!(scout.num_outputs, skink.num_outputs);
    assert_eq!(
        scout.blueprint.modules.len(),
        skink.blueprint.modules.len() + 1
    );
    assert_eq!(count_sensors(&skink.blueprint, SensorType::Ultrasonic), 0);
    // The stalk's far end lies outside the head: further from the head's
    // centre than its radius.
    let spawned = bevy::math::Quat::from_rotation_x(-std::f32::consts::PI);
    let head = scout
        .blueprint
        .root_module
        .and_then(|id| scout.blueprint.modules.get(&id));
    let head = head.expect("the scout has a head");
    let stalk = scout
        .blueprint
        .modules
        .values()
        .find(|m| {
            m.sensors
                .iter()
                .any(|s| s.sensor_type == SensorType::Ultrasonic)
        })
        .expect("the eye is mounted on a part");
    let eye = &stalk.sensors[0];
    let eye_at = stalk.transform.0 + stalk.transform.1 * eye.local_position;
    let reach = spawned * (eye_at - head.transform.0);
    assert!(
        reach.length() > 0.52 + 0.1 && reach.x > 0.5,
        "the eye sits at {reach}"
    );

    let eel = body(MORPHOLOGY_POOL[2], INITIAL_LSYSTEM_ITERATIONS);
    assert_eq!(slider_ranges(&eel.blueprint), vec![(-0.2, 0.2)]);
}

/// What breeding's body mutation makes of each plan, over 300 mutants: few
/// are lost (no body, or too few motors to be scored) and a fair share come
/// out with another build (measured: 4 percent lost, 10 to 13 rebuilt). Six
/// of the former catalogue's eight walkers changed their build in under one
/// mutant in a hundred.
#[test]
fn mutation_rebuilds_a_plan_without_losing_it() {
    const MUTANTS: usize = 300;
    for (name, source) in plans() {
        let mut rng = Pcg64::seed_from_u64(7);
        let parent = genotype(source, &mut rng);
        let before = scorable(&parent).expect("the plan is scorable");
        let (mut lost, mut rebuilt) = (0, 0);
        for _ in 0..MUTANTS {
            let mut child = parent.clone();
            child.mutate_body(&mut rng, 0.3);
            match scorable(&child) {
                None => lost += 1,
                Some(after) => rebuilt += usize::from(build_of(&after) != build_of(&before)),
            }
        }
        assert!(
            lost * 100 <= MUTANTS * 8,
            "{name}: {lost} of {MUTANTS} mutants lost"
        );
        assert!(
            rebuilt * 100 >= MUTANTS * 6,
            "{name}: only {rebuilt} of {MUTANTS} rebuilt"
        );
    }
}

/// Every plan grows the same legs and trunk from the same rules, so a child
/// of any two plans is one of the family: a body the lab can score, on at
/// least four stiff legs. (A plan whose legs had knees took over the gene
/// pool it was tried in and held it at a third of the fitness.)
#[test]
fn any_two_plans_cross_into_one_of_the_family() {
    let mut rng = Pcg64::seed_from_u64(13);
    for (a_name, a) in plans() {
        for (b_name, b) in plans() {
            let (pa, pb) = (genotype(a, &mut rng), genotype(b, &mut rng));
            for _ in 0..10 {
                let child = pa.body_child(&pb, 0.0, &mut rng);
                let Some(grown) = scorable(&child) else {
                    panic!("{a_name} x {b_name}: the child is lost");
                };
                assert!(
                    feet(&grown) >= 4,
                    "{a_name} x {b_name}: {} feet",
                    feet(&grown)
                );
                assert_eq!(kneed_legs(&grown.blueprint), 0, "{a_name} x {b_name}");
            }
        }
    }
    // An eel's short links averaged into a skink's lengthen the child's
    // trunk past its threshold: a third trunk segment.
    let (skink, eel) = (
        genotype(MORPHOLOGY_POOL[0], &mut rng),
        genotype(MORPHOLOGY_POOL[2], &mut rng),
    );
    let child = scorable(&skink.body_child(&eel, 0.0, &mut rng)).expect("scorable");
    let skink = scorable(&skink).expect("scorable");
    assert_eq!(
        child.blueprint.modules.len(),
        skink.blueprint.modules.len() + 1
    );
}

/// The plan the desktop app loads from `assets/morphologies/` joins the
/// gene pool there, so it starts as the skink: the same body, held to what
/// the plans are held to.
#[test]
fn the_example_plan_starts_as_the_skink() {
    let example = include_str!("../assets/morphologies/example_walker.lsys");
    let (ours, skink) = (
        body(example, INITIAL_LSYSTEM_ITERATIONS),
        body(MORPHOLOGY_POOL[0], INITIAL_LSYSTEM_ITERATIONS),
    );
    assert_eq!(ours.blueprint.modules.len(), skink.blueprint.modules.len());
    assert_eq!(ours.num_outputs, skink.num_outputs);
    let mass = |derived: &DerivedBody| {
        derived
            .blueprint
            .modules
            .values()
            .map(|m| m.mass)
            .sum::<f32>()
    };
    assert!((mass(&ours) - mass(&skink)).abs() < 1e-3);
    let rules = |source: &str| {
        let system = System::from_source(source).expect("parses").to_source();
        system
            .lines()
            .filter(|line| !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(rules(example), rules(MORPHOLOGY_POOL[0]));
}

#[test]
fn mutated_morphology_still_parses() {
    use symbios::system::mutate::MutationConfig;

    let mut rng = Pcg64::seed_from_u64(42);

    let mut system = System::from_source(MORPHOLOGY_POOL[0]).expect("valid source");
    let config = MutationConfig {
        constant_rate: 0.5,
        constant_strength: 0.3,
        gaussian_jitter_scale: 0.3,
        gaussian_jitter_rate: 0.5,
        ..MutationConfig::default()
    };

    // Run mutation 10 times and verify each result is still parseable
    for i in 0..10 {
        system.mutate_with_rng(&mut rng, &config);
        let source = system.to_source();
        let result = System::from_source(&source);
        assert!(
            result.is_ok(),
            "Mutation round {} produced unparseable source: {}",
            i,
            source
        );
    }
}
