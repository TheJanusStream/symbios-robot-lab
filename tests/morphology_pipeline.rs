//! Integration tests for the L-System morphology → RobotBlueprint pipeline.
//!
//! These tests verify that L-System source strings produce valid robot
//! topologies with the expected number of modules, joints, and joint types.

use symbios::System;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter};
use symbios_robot_lab::core::presets::MORPHOLOGY_POOL;

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
// Mutation stability
// ---------------------------------------------------------------------------

const MILLIPEDE: &str = MORPHOLOGY_POOL[0];

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
    use symbios_robot::SensorType;
    let bp = build_blueprint_from_source("omega: B(1) Su");
    assert_eq!(bp.modules.len(), 1);
    let sensors: Vec<_> = bp
        .modules
        .values()
        .flat_map(|m| m.sensors.iter().map(|s| s.sensor_type))
        .collect();
    assert!(
        sensors.iter().any(|t| matches!(t, SensorType::Ultrasonic)),
        "Expected an Ultrasonic sensor, got {:?}",
        sensors
    );
}

#[test]
fn every_preset_parses_and_has_a_walker_topology() {
    // After the v0.2 preset overhaul, every entry must be a credible
    // legged walker: parses, spawns ≥ 2 modules and ≥ 1 hinge joint,
    // mounts an IMU on the body, and ends every foot with a touch
    // sensor. Touch sensors are required for `FootAirTimeFactor` and
    // `ComBounceReward` to score real step cycles; a preset without
    // them slips past the anti-vibrobot fitness terms.
    use symbios_robot::SensorType;
    for (idx, source) in MORPHOLOGY_POOL.iter().enumerate() {
        let bp = build_blueprint_from_source(source);
        assert!(
            bp.modules.len() >= 2,
            "preset[{idx}] produced only {} module(s)",
            bp.modules.len()
        );
        assert!(
            count_hinge_joints(&bp) >= 1,
            "preset[{idx}] produced no hinge joints",
        );
        let imus = bp
            .modules
            .values()
            .flat_map(|m| m.sensors.iter())
            .filter(|s| matches!(s.sensor_type, SensorType::IMU))
            .count();
        assert!(imus >= 1, "preset[{idx}] has no IMU sensor (Si)");
        let touches = bp
            .modules
            .values()
            .flat_map(|m| m.sensors.iter())
            .filter(|s| matches!(s.sensor_type, SensorType::Touch))
            .count();
        assert!(
            touches >= 2,
            "preset[{idx}] has {touches} touch sensors (St); every foot must have one",
        );
    }
}

#[test]
fn preset_pool_holds_eight_walkers() {
    // Lock the pool size: Sentinel/Inchworm/Snake/Biped/Starfish were
    // removed in the v0.2 overhaul because they couldn't win against the
    // anti-vibrobot fitness, and Tripod, Strider and Scout joined to bring in
    // sliders, ball joints and distance sensing. This fails if the count
    // drifts.
    assert_eq!(MORPHOLOGY_POOL.len(), 8);
}

#[test]
fn the_new_presets_slide_stand_on_limited_ankles_and_see() {
    use symbios_robot::SensorType;
    let prismatic = |bp: &symbios_robot::RobotBlueprint| {
        bp.joints
            .iter()
            .filter(|j| matches!(j.joint_type, JointType::Prismatic { .. }))
            .count()
    };

    // Tripod: each of the three shins slides along its leg, within the
    // limit given for it (the last limit about the leg's axis).
    let tripod = build_blueprint_from_source(MORPHOLOGY_POOL[5]);
    assert_eq!(prismatic(&tripod), 3);
    for joint in &tripod.joints {
        if let JointType::Prismatic { axis } = joint.joint_type {
            let own = joint
                .limits
                .iter()
                .rev()
                .find(|l| l.axis.normalize().dot(axis.normalize()).abs() > 0.999)
                .expect("a slider has its own limit");
            assert_eq!((own.min, own.max), (-0.15, 0.15));
        }
    }

    // Strider: four ball-jointed ankles, each with exactly its own two
    // limits (the leg's hinge limits cleared).
    let strider = build_blueprint_from_source(MORPHOLOGY_POOL[6]);
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

    // Scout: one ultrasonic eye, looking forward (+X, the direction the
    // walker fitness rewards) and down once the robot is turned over to
    // spawn.
    let scout = build_blueprint_from_source(MORPHOLOGY_POOL[7]);
    let eyes: Vec<_> = scout
        .modules
        .values()
        .flat_map(|m| m.sensors.iter().map(move |s| (m, s)))
        .filter(|(_, s)| s.sensor_type == SensorType::Ultrasonic)
        .collect();
    assert_eq!(eyes.len(), 1);
    let (module, eye) = eyes[0];
    let spawned = bevy::math::Quat::from_rotation_x(-std::f32::consts::PI);
    let gaze = spawned * module.transform.1 * eye.local_rotation * bevy::math::Vec3::Y;
    assert!(gaze.x > 0.5 && gaze.y < -0.5, "the eye looks along {gaze}");
}

#[test]
fn mutated_morphology_still_parses() {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios::system::mutate::MutationConfig;

    let mut rng = Pcg64::seed_from_u64(42);

    let mut system = System::from_source(MILLIPEDE).expect("valid source");
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
