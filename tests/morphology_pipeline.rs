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
/// is the limb's count-down, whose flip grows every limb a segment per step.
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
/// constant mutation's reach) gives the skink a third trunk segment, and
/// with the step a fresh body has in hand its pair of legs and their feet
/// at once.
#[test]
fn a_longer_trunk_grows_a_third_pair_of_legs_at_once() {
    let skink = MORPHOLOGY_POOL[0];
    let before = body(skink, INITIAL_LSYSTEM_ITERATIONS);

    let longer = with_constant(skink, "trunk", 1.1);
    let grown = body(&longer, INITIAL_LSYSTEM_ITERATIONS);
    assert_eq!(
        grown.num_outputs,
        before.num_outputs + 3,
        "a trunk hinge and two hips"
    );
    assert_eq!(feet(&grown), feet(&before) + 2);
    assert_eq!(kneed_legs(&grown.blueprint), 0);

    // A tenth less changes nothing: the plan does not sit on a threshold.
    let shorter = body(
        &with_constant(skink, "trunk", 0.9),
        INITIAL_LSYSTEM_ITERATIONS,
    );
    assert_eq!(build_of(&shorter), build_of(&before));
}

/// The lengths of the leg capsules of a body, a leg at a time: the parts
/// from each foot up to the trunk.
fn leg_lengths(derived: &DerivedBody) -> Vec<Vec<f32>> {
    let bp = &derived.blueprint;
    let leg_of = legs(bp);
    let mut by_leg: std::collections::BTreeMap<usize, Vec<f32>> = Default::default();
    let mut ids: Vec<_> = bp.modules.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let (Some(&leg), Some(module)) = (leg_of.get(&id), bp.modules.get(&id)) else {
            continue;
        };
        if let symbios_robot::ShapePrimitive::Capsule { height, .. } = module.shape {
            by_leg.entry(leg).or_default().push(height);
        }
    }
    let footed: std::collections::BTreeSet<usize> = bp
        .modules
        .iter()
        .filter(|(_, m)| m.sensors.iter().any(|s| s.sensor_type == SensorType::Touch))
        .filter_map(|(id, _)| leg_of.get(id).copied())
        .collect();
    by_leg
        .into_iter()
        .filter(|(leg, _)| footed.contains(leg))
        .map(|(_, lengths)| lengths)
        .collect()
}

/// The ranges (lowest, highest) of the hinges on the legs that end in a
/// foot, hips first then knees, a leg at a time.
fn leg_hinge_ranges(derived: &DerivedBody) -> Vec<Vec<(f32, f32)>> {
    let bp = &derived.blueprint;
    let leg_of = legs(bp);
    let footed: std::collections::BTreeSet<usize> = bp
        .modules
        .iter()
        .filter(|(_, m)| m.sensors.iter().any(|s| s.sensor_type == SensorType::Touch))
        .filter_map(|(id, _)| leg_of.get(id).copied())
        .collect();
    let mut by_leg: std::collections::BTreeMap<usize, Vec<(f32, f32)>> = Default::default();
    for joint in &bp.joints {
        let Some(&leg) = leg_of.get(&joint.child_id) else {
            continue;
        };
        if footed.contains(&leg)
            && let Some(limit) = joint.limits.last()
        {
            by_leg.entry(leg).or_default().push((limit.min, limit.max));
        }
    }
    by_leg.into_values().collect()
}

/// Whether every number of `got` is within a hundred-thousandth of `want`'s.
fn close(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-5)
}

/// The ranges of the hinges on the trunk (the parts on the body's midline),
/// lowest and highest.
fn trunk_hinge_ranges(derived: &DerivedBody) -> Vec<(f32, f32)> {
    let bp = &derived.blueprint;
    bp.joints
        .iter()
        .filter(|joint| matches!(joint.joint_type, JointType::Hinge { .. }))
        .filter(|joint| {
            bp.modules
                .get(&joint.child_id)
                .is_some_and(|m| m.transform.0.z.abs() < 1e-3)
        })
        .filter_map(|joint| joint.limits.last().map(|limit| (limit.min, limit.max)))
        .collect()
}

/// One mutated constant splits every leg: a tenth and a half more `segs`
/// (within one constant mutation's reach) and each of the skink's four legs
/// is two halves of its old length with a hinge between them, turning by
/// `bend` either way. Nothing else changes, and the body is whole at once.
#[test]
fn one_constant_splits_every_leg_in_two() {
    let skink = MORPHOLOGY_POOL[0];
    let before = body(skink, INITIAL_LSYSTEM_ITERATIONS);
    let stiff = leg_lengths(&before);
    assert_eq!(stiff.len(), 4);
    assert!(stiff.iter().all(|leg| close(leg, &[0.8])), "{stiff:?}");

    let split = body(
        &with_constant(skink, "segs", 1.15),
        INITIAL_LSYSTEM_ITERATIONS,
    );
    let halves = leg_lengths(&split);
    assert_eq!(halves.len(), 4);
    assert!(
        halves.iter().all(|leg| close(leg, &[0.4, 0.4])),
        "{halves:?}"
    );
    assert_eq!(kneed_legs(&split.blueprint), 4);
    assert_eq!(split.num_outputs, before.num_outputs + 4);
    assert_eq!(feet(&split), feet(&before));
    for ranges in leg_hinge_ranges(&split) {
        assert_eq!(ranges.len(), 2, "a hip and a knee");
        let (hip, knee) = (ranges[0], ranges[1]);
        assert!(
            (hip.0 + 0.7).abs() < 1e-5 && (hip.1 - 0.7).abs() < 1e-5,
            "{hip:?}"
        );
        assert!(
            (knee.0 + 0.1).abs() < 1e-5 && (knee.1 - 0.1).abs() < 1e-5,
            "{knee:?}"
        );
    }
    // The knee's range is a constant of its own.
    let wider = with_constant(&with_constant(skink, "segs", 1.15), "bend", 4.0);
    for ranges in leg_hinge_ranges(&body(&wider, INITIAL_LSYSTEM_ITERATIONS)) {
        assert!((ranges[1].1 - 0.4).abs() < 1e-5, "{ranges:?}");
    }

    // A tenth less changes nothing; twice as much splits each leg in three,
    // which takes the body one step more to finish.
    let fewer = body(
        &with_constant(skink, "segs", 0.9),
        INITIAL_LSYSTEM_ITERATIONS,
    );
    assert_eq!(build_of(&fewer), build_of(&before));
    let thrice = body(
        &with_constant(skink, "segs", 2.0),
        INITIAL_LSYSTEM_ITERATIONS + 1,
    );
    for lengths in leg_lengths(&thrice) {
        assert_eq!(lengths.len(), 3);
        assert!(
            lengths.iter().all(|l| (l - 0.8 / 3.0).abs() < 1e-5),
            "{lengths:?}"
        );
    }
    assert_eq!(leg_lengths(&thrice).len(), 4);
}

/// The axiom bounds what the physics cannot bear and what a body could not
/// finish: however far a constant drifts, the trunk is no thinner than
/// 0.3 m, a hip turns no further than 0.75 rad either way, the trunk's
/// joints 1.2 rad, a knee 1 rad, and a leg has three segments at most.
#[test]
fn the_axiom_bounds_what_a_constant_may_do() {
    let skink = MORPHOLOGY_POOL[0];
    let trunk_radii = |source: &str| -> Vec<f32> {
        body(source, INITIAL_LSYSTEM_ITERATIONS)
            .blueprint
            .modules
            .values()
            .filter(|m| m.transform.0.z.abs() < 1e-3)
            .filter_map(|m| match m.shape {
                symbios_robot::ShapePrimitive::Capsule { radius, .. } => Some(radius),
                _ => None,
            })
            .collect()
    };
    assert!(trunk_radii(skink).iter().all(|r| (r - 0.36).abs() < 1e-5));
    let thin = with_constant(skink, "girth", 0.5);
    assert!(trunk_radii(&thin).iter().all(|r| (r - 0.3).abs() < 1e-5));

    let wide = with_constant(&with_constant(skink, "fore", 3.0), "aft", 3.0);
    for ranges in leg_hinge_ranges(&body(&wide, INITIAL_LSYSTEM_ITERATIONS)) {
        assert_eq!(ranges.len(), 1);
        assert!(
            close(&[ranges[0].0, ranges[0].1], &[-0.75, 0.75]),
            "{ranges:?}"
        );
    }
    let loose = with_constant(&with_constant(skink, "segs", 1.15), "bend", 30.0);
    for ranges in leg_hinge_ranges(&body(&loose, INITIAL_LSYSTEM_ITERATIONS)) {
        assert!(
            close(&[ranges[1].0, ranges[1].1], &[-1.0, 1.0]),
            "{ranges:?}"
        );
    }
    let before = trunk_hinge_ranges(&body(skink, INITIAL_LSYSTEM_ITERATIONS));
    assert_eq!(before.len(), 2);
    assert!(
        before.iter().all(|r| close(&[r.0, r.1], &[-0.5, 0.5])),
        "{before:?}"
    );
    let bendy = with_constant(&with_constant(skink, "yawl", 5.0), "yawr", 4.0);
    let after = trunk_hinge_ranges(&body(&bendy, INITIAL_LSYSTEM_ITERATIONS));
    assert!(
        after.iter().all(|r| close(&[r.0, r.1], &[-1.2, 1.2])),
        "{after:?}"
    );

    // A leg has three segments at most, and one at least.
    let many = with_constant(skink, "segs", 9.0);
    let lengths = leg_lengths(&body(&many, INITIAL_LSYSTEM_ITERATIONS + 1));
    assert_eq!(lengths.len(), 4);
    assert!(lengths.iter().all(|leg| leg.len() == 3), "{lengths:?}");
    let none = with_constant(skink, "segs", 0.1);
    let lengths = leg_lengths(&body(&none, INITIAL_LSYSTEM_ITERATIONS));
    assert_eq!(lengths.len(), 4);
    assert!(lengths.iter().all(|leg| close(leg, &[0.8])), "{lengths:?}");
}

/// A fresh body has a step in hand: every plan is whole a step before the
/// lab's starting count, and a step before that still lacks its hind feet's
/// sensors (they arrive a step after the leg they end).
#[test]
fn every_plan_is_whole_a_step_before_the_lab_derives_it() {
    for (name, source) in plans() {
        let ours = body(source, INITIAL_LSYSTEM_ITERATIONS);
        let earlier = body(source, INITIAL_LSYSTEM_ITERATIONS - 1);
        assert_eq!(build_of(&earlier), build_of(&ours), "{name}");
        let too_early = body(source, INITIAL_LSYSTEM_ITERATIONS - 2);
        assert_eq!(too_early.num_outputs, ours.num_outputs, "{name}");
        assert_eq!(feet(&too_early) + 2, feet(&ours), "{name}");
    }
}

/// A condition holds one comparison: the L-system writes an AND back in a
/// form it cannot read, and the lab stores bodies as written back.
#[test]
fn a_plans_conditions_hold_one_comparison() {
    for (name, source) in plans() {
        let system = System::from_source(source).expect("the plan parses");
        for rule in system.rules.values().flatten() {
            let ops = rule.condition.iter().flatten();
            let comparisons = ops
                .clone()
                .filter(|op| matches!(op, Op::Gt | Op::Lt | Op::Ge | Op::Le | Op::Eq | Op::Ne))
                .count();
            assert_eq!(
                comparisons, 1,
                "{name}: a condition compares {comparisons} times"
            );
            assert!(
                !ops.clone()
                    .any(|op| matches!(op, Op::And | Op::Or | Op::Not)),
                "{name}: a condition joins comparisons"
            );
        }
        // And the plan is what the lab stores: written back and read again,
        // it is the same text.
        let stored = system.to_source();
        let again = System::from_source(&stored).expect("the stored plan parses");
        assert_eq!(again.to_source(), stored, "{name}");
    }
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
/// out with another build (measured over 4000: 2 percent lost, 13 to 16
/// rebuilt, of which 7 a third pair of legs and 3 a knee in every leg). Six
/// of the first catalogue's eight walkers changed their build in under one
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

/// Every plan grows its limbs by the same two rules, so a child of any two
/// plans is one of the family: a body the lab can score, on at least four
/// stiff legs.
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
}

/// A child takes each constant whole from one parent or the other, so what
/// a threshold grew is inherited or it is not: of the children of a skink
/// whose legs have split and one whose legs have not, about half have four
/// knees and the rest none. (Averaged, as constants once were, `segs` came
/// out at 1.45 in every child and no child had a knee.)
#[test]
fn a_child_inherits_a_knee_whole_or_not_at_all() {
    let mut rng = Pcg64::seed_from_u64(17);
    let plain = genotype(MORPHOLOGY_POOL[0], &mut rng);
    let split = genotype(&with_constant(MORPHOLOGY_POOL[0], "segs", 1.15), &mut rng);
    let mut kneed = 0;
    for pair in [(&split, &plain), (&plain, &split)] {
        for _ in 0..40 {
            let child = pair.0.body_child(pair.1, 0.0, &mut rng);
            let grown = scorable(&child).expect("the child is scorable");
            let knees = kneed_legs(&grown.blueprint);
            assert!(knees == 0 || knees == 4, "{knees} knees");
            kneed += usize::from(knees == 4);
        }
    }
    assert!((25..=55).contains(&kneed), "{kneed} of 80 children kneed");
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
