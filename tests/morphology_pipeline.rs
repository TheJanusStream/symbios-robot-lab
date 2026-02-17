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
        .filter(|j| j.joint_type == JointType::Hinge)
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

const QUADRUPED: &str = MORPHOLOGY_POOL[0];

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
