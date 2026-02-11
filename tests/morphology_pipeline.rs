//! Integration tests for the L-System morphology → RobotBlueprint pipeline.
//!
//! These tests verify that L-System source strings produce valid robot
//! topologies with the expected number of modules, joints, and joint types.

use symbios::System;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter};

/// Parse an L-System morphology string and build a RobotBlueprint.
fn build_blueprint_from_source(source: &str) -> symbios_robot::RobotBlueprint {
    let mut sys = System::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("omega:") {
            let axiom = trimmed.trim_start_matches("omega:").trim();
            sys.set_axiom(axiom).expect("valid axiom");
        } else if trimmed.contains("->") {
            sys.add_rule(trimmed).expect("valid rule");
        }
    }
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
// Default quadruped morphology (from lifecycle.rs)
// ---------------------------------------------------------------------------

const QUADRUPED: &str = "\
omega: !(0.6) B(0.3, 0.6, 0.6) \
[ &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ \\(90) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ |(1) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ /(90) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ]";

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
    assert!(bp.root_module.is_some(), "Blueprint should have a root module");
}

#[test]
fn quadruped_joint_count_matches_lifecycle_constant() {
    let bp = build_blueprint_from_source(QUADRUPED);
    let hinge_count = count_hinge_joints(&bp);
    // This must match DEFAULT_NUM_JOINTS in lifecycle.rs (8)
    // which is used to size the NEAT output layer.
    assert_eq!(hinge_count, 8, "Hinge count must match DEFAULT_NUM_JOINTS");
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
