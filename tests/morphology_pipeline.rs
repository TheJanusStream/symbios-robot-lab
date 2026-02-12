//! Integration tests for the L-System morphology → RobotBlueprint pipeline.
//!
//! These tests verify that L-System source strings produce valid robot
//! topologies with the expected number of modules, joints, and joint types.

use symbios::System;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter, SensorType};

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
        .filter(|j| j.joint_type == JointType::Hinge)
        .count()
}

/// Count sensors of a given type across all modules.
fn count_sensors(bp: &symbios_robot::RobotBlueprint, sensor_type: SensorType) -> usize {
    bp.modules
        .values()
        .flat_map(|m| &m.sensors)
        .filter(|s| s.sensor_type == sensor_type)
        .count()
}

// ---------------------------------------------------------------------------
// Default quadruped morphology (mirrors lifecycle.rs DEFAULT_MORPHOLOGY)
// ---------------------------------------------------------------------------

const QUADRUPED: &str = "\
#define hip_min -1.57
#define hip_max 1.57
#define hip_eff 50
#define hip_vel 5
#define knee_min -2.0
#define knee_max 0.2
#define knee_eff 30
#define knee_vel 5
#define lw 0.15
#define seg 0.5
omega: !(0.6) K(0.2, 0.3) Si [ &(90) L ] [ \\(90) &(90) L ] [ |(1) &(90) L ] [ /(90) &(90) L ]
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) B(seg) Jl(knee_min, knee_max, knee_eff, knee_vel) J B(seg) St";

#[test]
fn quadruped_has_correct_module_count() {
    let bp = build_blueprint_from_source(QUADRUPED);
    // 1 body + 4 legs × 2 segments = 9 modules
    assert_eq!(bp.modules.len(), 9, "Quadruped should have 9 modules");
}

#[test]
fn quadruped_has_correct_joint_count() {
    let bp = build_blueprint_from_source(QUADRUPED);
    // 4 legs × 2 hinge joints = 8 joints
    assert_eq!(bp.joints.len(), 8, "Quadruped should have 8 joints");
}

#[test]
fn quadruped_all_joints_are_hinge() {
    let bp = build_blueprint_from_source(QUADRUPED);
    let hinge_count = count_hinge_joints(&bp);
    assert_eq!(hinge_count, 8, "All 8 joints should be hinge type");
}

#[test]
fn quadruped_joints_have_limits() {
    let bp = build_blueprint_from_source(QUADRUPED);
    for (i, joint) in bp.joints.iter().enumerate() {
        assert!(
            joint.limits.is_some(),
            "Joint {} should have limits set via Jl",
            i
        );
    }
}

#[test]
fn quadruped_has_root_module() {
    let bp = build_blueprint_from_source(QUADRUPED);
    assert!(
        bp.root_module.is_some(),
        "Blueprint should have a root module"
    );
}

#[test]
fn quadruped_joint_count_matches_dynamic_derivation() {
    let bp = build_blueprint_from_source(QUADRUPED);
    let hinge_count = count_hinge_joints(&bp);
    // derive_brain_topology dynamically counts hinge joints for NEAT output sizing.
    assert_eq!(hinge_count, 8, "Quadruped should have 8 hinge joints");
}

#[test]
fn quadruped_has_imu_sensor() {
    let bp = build_blueprint_from_source(QUADRUPED);
    assert_eq!(
        count_sensors(&bp, SensorType::IMU),
        1,
        "Quadruped should have 1 IMU on the body"
    );
}

#[test]
fn quadruped_has_touch_sensors() {
    let bp = build_blueprint_from_source(QUADRUPED);
    assert_eq!(
        count_sensors(&bp, SensorType::Touch),
        4,
        "Quadruped should have 4 touch sensors (one per foot)"
    );
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
// Morphology pool: all initial morphologies must parse and produce valid robots
// ---------------------------------------------------------------------------

const CENTIPEDE: &str = "\
#define jmin -1.2
#define jmax 1.2
#define eff 40
#define vel 5
#define lw 0.12
#define seg 0.4
omega: !(0.4) K(0.15, 0.25) Si S
S -> Jl(jmin, jmax, eff, vel) J B(seg) [ \\(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) B(seg) St ] [ /(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) B(seg) St ] S";

const HEXAPOD: &str = "\
#define jmin -1.57
#define jmax 1.57
#define eff 40
#define vel 5
#define lw 0.12
#define seg 0.4
omega: !(0.5) K(0.2, 0.35) Si [ &(90) L ] [ \\(60) &(90) L ] [ \\(120) &(90) L ] [ |(1) &(90) L ] [ /(120) &(90) L ] [ /(60) &(90) L ]
L -> Jl(jmin, jmax, eff, vel) J !(lw) B(seg) Jl(jmin, jmax, eff, vel) J B(seg) St";

const SNAKE: &str = "\
#define jmin -0.8
#define jmax 0.8
#define eff 30
#define vel 8
#define seg 0.35
omega: !(0.3) K(0.15, 0.2) Si St S
S -> Jl(jmin, jmax, eff, vel) J B(seg) St S";

const BIPED: &str = "\
#define hip_min -1.57
#define hip_max 1.57
#define eff 60
#define vel 5
#define knee_min -2.0
#define knee_max 0.2
#define keff 40
#define kvel 5
#define lw 0.18
#define seg 0.6
omega: !(0.5) K(0.25, 0.4) Si [ &(90) L ] [ |(1) &(90) L ]
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) B(seg) Jl(knee_min, knee_max, keff, kvel) J B(seg) St";

const STARFISH: &str = "\
#define jmin -1.0
#define jmax 1.0
#define eff 35
#define vel 5
#define seg 0.5
omega: !(0.5) K(0.2, 0.2) Si [ &(90) A ] [ \\(72) &(90) A ] [ \\(144) &(90) A ] [ /(144) &(90) A ] [ /(72) &(90) A ]
A -> Jl(jmin, jmax, eff, vel) J B(seg) St";

/// Validate that every morphology in the pool parses, derives, and produces
/// a non-trivial robot with at least 1 module, 1 joint, and 1 sensor.
#[test]
fn morphology_pool_all_valid() {
    let pool = [
        ("quadruped", QUADRUPED),
        ("centipede", CENTIPEDE),
        ("hexapod", HEXAPOD),
        ("snake", SNAKE),
        ("biped", BIPED),
        ("starfish", STARFISH),
    ];

    for (name, source) in &pool {
        let bp = build_blueprint_from_source(source);
        let hinges = count_hinge_joints(&bp);
        let imus = count_sensors(&bp, SensorType::IMU);
        let touches = count_sensors(&bp, SensorType::Touch);

        assert!(
            bp.modules.len() >= 2,
            "{}: expected at least 2 modules, got {}",
            name,
            bp.modules.len()
        );
        assert!(
            hinges >= 1,
            "{}: expected at least 1 hinge joint, got {}",
            name,
            hinges
        );
        assert!(
            imus >= 1,
            "{}: expected at least 1 IMU, got {}",
            name,
            imus
        );
        assert!(
            touches >= 1,
            "{}: expected at least 1 touch sensor, got {}",
            name,
            touches
        );

        println!(
            "  {}: {} modules, {} hinges, {} IMUs, {} touch",
            name,
            bp.modules.len(),
            hinges,
            imus,
            touches
        );
    }
}

#[test]
fn centipede_has_recursive_segments() {
    let bp = build_blueprint_from_source(CENTIPEDE);
    // 3 derivation steps → 3 spine segments + 3×2 leg segments + 1 body = 10 modules
    // Each spine segment has 1 spine joint + 2 leg joints = 3 hinges per segment = 9 total
    assert!(bp.modules.len() >= 7, "Centipede should have multiple segments");
    assert!(count_hinge_joints(&bp) >= 6, "Centipede should have multiple joints");
}

#[test]
fn hexapod_has_six_legs() {
    let bp = build_blueprint_from_source(HEXAPOD);
    assert_eq!(
        count_sensors(&bp, SensorType::Touch),
        6,
        "Hexapod should have 6 touch sensors (one per foot)"
    );
    // 6 legs × 2 joints = 12
    assert_eq!(count_hinge_joints(&bp), 12, "Hexapod should have 12 hinge joints");
}

#[test]
fn biped_has_two_legs() {
    let bp = build_blueprint_from_source(BIPED);
    assert_eq!(
        count_sensors(&bp, SensorType::Touch),
        2,
        "Biped should have 2 touch sensors"
    );
    assert_eq!(count_hinge_joints(&bp), 4, "Biped should have 4 hinge joints");
}

#[test]
fn starfish_has_five_arms() {
    let bp = build_blueprint_from_source(STARFISH);
    assert_eq!(
        count_sensors(&bp, SensorType::Touch),
        5,
        "Starfish should have 5 touch sensors"
    );
    assert_eq!(count_hinge_joints(&bp), 5, "Starfish should have 5 hinge joints");
}

// ---------------------------------------------------------------------------
// Mutation stability
// ---------------------------------------------------------------------------

#[test]
fn mutated_morphology_still_parses() {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios::system::mutate::MutationConfig;

    let mut rng = Pcg64::seed_from_u64(42);

    let mut system = System::from_source(QUADRUPED).expect("valid source");
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
