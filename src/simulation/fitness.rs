//! Fitness scoring for evolved walkers: pure functions of a robot's measured
//! trajectory, composed from `symbios-genetics` scorers.
//!
//! The walker fitness is forward displacement times seven quality factors in
//! `[0, 1]`, an anti-"vibrobot" cocktail added after the plain
//! `displacement × up_alignment × (height + 0.5)` formula was reward-hacked
//! by tiny high-frequency joint oscillations sliding across the floor:
//!
//! | Factor | Catches |
//! |--------|---------|
//! | `OrientationStability` (1 − stddev/0.4) | tipping, wobbling |
//! | `SmoothnessReward` (1/(1+Σ Δv²/N)) | controller chatter |
//! | `ControlJerkReward` (1/(1+Σ ΔΔv²/(N·s))) | sign-flipping vibration |
//! | `JerkReward` (1/(1+body_jerk/scale)) | flailing body motion |
//! | `VelocityGate` (clamp v/0.2) | slow vibrobot sliding |
//! | `FootAirTimeFactor` (transitions/target) | no real step cycles |
//! | `ComBounceReward` (clamp σ_y/0.02) | flat sliding (no bounce) |
//!
//! Energy is estimated from actual joint angle change per tick
//! (`Σ |τ_max · Δθ|`), feeding the biomechanics Cost of Transport
//! `E / (m·g·d)`, NSGA-II's second objective.

use crate::core::population::Agent;
use crate::simulation::physics::GRAVITY;
use symbios_genetics::scorers::{Multiply, Scorer};

// -- Scoring scales --------------------------------------------------------
//
// These constants encode "how bad is bad" for each penalty. They were chosen
// so a competent real walker scores ~0.7+ on each factor and a vibrobot
// scores ~0.1- on the targeted factor, while leaving headroom on either side.

/// Stddev(up_alignment) at which `OrientationStability` reaches 0.
const ORIENT_STDDEV_SCALE: f32 = 0.4;
/// Per-tick-per-joint mean Δv² scale for `SmoothnessReward`.
const SMOOTHNESS_SCALE: f32 = 1.0;
/// Per-tick-per-joint mean ΔΔv² scale for `ControlJerkReward`: a walker doing
/// a 2 Hz sinusoid scores ~0.8, a ±ε vibrator ~0.1 even for tiny ε.
const CONTROL_JERK_SCALE: f32 = 0.025;
/// Per-tick body jerk² scale; healthy walkers stay ~0.5+.
const BODY_JERK_SCALE: f32 = 50.0;
/// Average forward velocity (m/s) at which `VelocityGate` saturates.
const VELOCITY_GATE_TARGET: f32 = 0.2;
/// Foot contact/air transitions per touch sensor at which
/// `FootAirTimeFactor` saturates.
pub const FOOT_TRANSITIONS_TARGET: f32 = 3.0;
/// Stddev(body_y) at which `ComBounceReward` saturates: 2 cm of vertical
/// motion is the floor, anything less is "sliding".
const COM_BOUNCE_TARGET: f32 = 0.02;
/// Mean commanded joint speed that maps to the top of the QD descriptor's
/// action-intensity axis.
const ACTION_INTENSITY_NORM: f32 = 5.0;

/// A robot's measured trajectory over the evaluation window: the input `S`
/// of the `Scorer<S, f32>` composition.
#[derive(Clone, Debug, Default)]
pub struct AgentTrajectory {
    /// Forward displacement projected on the initial forward direction (m).
    pub forward_displacement: f32,
    /// Average forward velocity over the window (m/s).
    pub avg_velocity: f32,
    /// Mean of `dot(local_up, world_up)`, clamped `[0, 1]`.
    pub avg_up_alignment: f32,
    /// Mean body height (m), for the inspector.
    pub avg_height: f32,
    /// Standard deviation of `up_alignment`.
    pub up_alignment_stddev: f32,
    /// Standard deviation of body height (m): the CoM "bounce".
    pub com_bounce: f32,
    /// Mechanical-work proxy `Σ |τ_max · Δθ|`, the numerator of CoT.
    pub mechanical_work: f32,
    /// Total robot mass (kg), the denominator of CoT.
    pub mass: f32,
    /// Mean per-tick per-joint `|Δv_target|²`.
    pub action_smoothness: f32,
    /// Mean per-tick per-joint `|a_t − 2·a_{t−1} + a_{t−2}|²`.
    pub control_jerk: f32,
    /// Mean per-tick `|Δv_body|² / Δt`.
    pub body_jerk: f32,
    /// Mean fraction of touch sensors in contact each tick. QD axis 0.
    pub support_duty_factor: f32,
    /// Contact/air transitions per touch sensor over the window.
    pub foot_transitions_per_sensor: f32,
    /// Mean per-joint commanded speed over the window. QD axis 1.
    pub action_intensity: f32,
}

/// Forward displacement (m), the headline distance signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardDisplacement;
impl Scorer<AgentTrajectory, f32> for ForwardDisplacement {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        t.forward_displacement
    }
}

/// `1 − stddev(up_alignment) / ORIENT_STDDEV_SCALE`, clamped `[0, 1]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrientationStability;
impl Scorer<AgentTrajectory, f32> for OrientationStability {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (1.0 - t.up_alignment_stddev / ORIENT_STDDEV_SCALE).clamp(0.0, 1.0)
    }
}

/// First-derivative smoothness: `1 / (1 + action_smoothness/scale)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmoothnessReward;
impl Scorer<AgentTrajectory, f32> for SmoothnessReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.action_smoothness.max(0.0) / SMOOTHNESS_SCALE)
    }
}

/// Second-derivative penalty `1 / (1 + control_jerk/scale)`: a ±ε
/// sign-flipping output has |ΔΔv| = 4ε per tick however small ε is.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlJerkReward;
impl Scorer<AgentTrajectory, f32> for ControlJerkReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.control_jerk.max(0.0) / CONTROL_JERK_SCALE)
    }
}

/// Body-jerk reward `1 / (1 + body_jerk/scale)`: flailing limbs jerk the torso.
#[derive(Clone, Copy, Debug, Default)]
pub struct JerkReward;
impl Scorer<AgentTrajectory, f32> for JerkReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.body_jerk.max(0.0) / BODY_JERK_SCALE)
    }
}

/// Minimum-velocity gate `(avg_velocity / target).clamp(0, 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityGate;
impl Scorer<AgentTrajectory, f32> for VelocityGate {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (t.avg_velocity / VELOCITY_GATE_TARGET).clamp(0.0, 1.0)
    }
}

/// Foot air-time factor `(transitions_per_sensor / target).clamp(0, 1)`. A
/// body with no touch sensors is given the target value when its trajectory
/// is built, so it is not punished for having no feet.
#[derive(Clone, Copy, Debug, Default)]
pub struct FootAirTimeFactor;
impl Scorer<AgentTrajectory, f32> for FootAirTimeFactor {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (t.foot_transitions_per_sensor / FOOT_TRANSITIONS_TARGET).clamp(0.0, 1.0)
    }
}

/// CoM vertical bounce `(stddev(body_y) / target).clamp(0, 1)`: walking bobs,
/// sliding does not.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComBounceReward;
impl Scorer<AgentTrajectory, f32> for ComBounceReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (t.com_bounce / COM_BOUNCE_TARGET).clamp(0.0, 1.0)
    }
}

/// Cost of Transport `E / (m · g · d)`, lower is better; infinite when the
/// robot did not progress, so standing still earns no fake efficiency.
#[derive(Clone, Copy, Debug, Default)]
pub struct CostOfTransport;
impl Scorer<AgentTrajectory, f32> for CostOfTransport {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        let denom = t.mass * GRAVITY * t.forward_displacement.max(0.0);
        if denom > f32::EPSILON {
            t.mechanical_work / denom
        } else {
            f32::INFINITY
        }
    }
}

/// CoT as a maximize-me reward in `(0, 1]`: `1 / (1 + CoT)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct InverseCoT;
impl Scorer<AgentTrajectory, f32> for InverseCoT {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        let cot = CostOfTransport.score(t);
        if cot.is_finite() {
            1.0 / (1.0 + cot.max(0.0))
        } else {
            0.0
        }
    }
}

type WalkerFitness = Multiply<
    Multiply<
        Multiply<
            Multiply<
                Multiply<
                    Multiply<Multiply<ForwardDisplacement, OrientationStability>, SmoothnessReward>,
                    ControlJerkReward,
                >,
                JerkReward,
            >,
            VelocityGate,
        >,
        FootAirTimeFactor,
    >,
    ComBounceReward,
>;

/// The walker fitness composite (see the module docs).
pub const fn walker_fitness() -> WalkerFitness {
    Multiply(
        Multiply(
            Multiply(
                Multiply(
                    Multiply(
                        Multiply(
                            Multiply(ForwardDisplacement, OrientationStability),
                            SmoothnessReward,
                        ),
                        ControlJerkReward,
                    ),
                    JerkReward,
                ),
                VelocityGate,
            ),
            FootAirTimeFactor,
        ),
        ComBounceReward,
    )
}

/// `(fitness, objectives, descriptor)` for a trajectory:
/// - **fitness**: the walker composite;
/// - **objectives**: `[walker_fitness, inverse_CoT]` for NSGA-II;
/// - **descriptor**: `[support_duty_factor, action_intensity_norm]` for QD.
pub fn evaluate_trajectory(t: &AgentTrajectory) -> (f32, Vec<f32>, Vec<f32>) {
    let fitness = walker_fitness().score(t);
    let objectives = vec![fitness, InverseCoT.score(t)];
    let descriptor = vec![
        t.support_duty_factor.clamp(0.0, 1.0),
        (t.action_intensity / ACTION_INTENSITY_NORM).clamp(0.0, 1.0),
    ];
    (fitness, objectives, descriptor)
}

/// The trajectory an agent's recorded metrics describe.
pub fn trajectory_of(agent: &Agent) -> AgentTrajectory {
    AgentTrajectory {
        forward_displacement: agent.displacement,
        avg_velocity: agent.avg_velocity,
        avg_up_alignment: agent.avg_up_alignment,
        avg_height: agent.avg_height,
        up_alignment_stddev: agent.up_alignment_stddev,
        com_bounce: agent.com_bounce,
        mechanical_work: agent.mechanical_work,
        mass: agent.mass.max(f32::EPSILON),
        action_smoothness: agent.action_smoothness,
        control_jerk: agent.control_jerk,
        body_jerk: agent.body_jerk,
        support_duty_factor: agent.support_duty_factor,
        foot_transitions_per_sensor: agent.foot_transitions_per_sensor,
        action_intensity: agent.action_intensity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_traj() -> AgentTrajectory {
        AgentTrajectory {
            forward_displacement: 1.0,
            avg_velocity: 0.5,
            avg_up_alignment: 0.9,
            avg_height: 0.5,
            up_alignment_stddev: 0.1,
            com_bounce: 0.05,
            mechanical_work: 5.0,
            mass: 2.0,
            action_smoothness: 0.2,
            control_jerk: 0.005,
            body_jerk: 5.0,
            support_duty_factor: 0.5,
            foot_transitions_per_sensor: 3.0,
            action_intensity: 1.0,
        }
    }

    #[test]
    fn cost_of_transport_matches_definition() {
        // CoT = E / (m·g·d). 100 J of work, 10 kg, 5 m → 100/(10·9.81·5) ≈ 0.2039.
        let mut t = empty_traj();
        t.mechanical_work = 100.0;
        t.mass = 10.0;
        t.forward_displacement = 5.0;
        let cot = CostOfTransport.score(&t);
        let expected = 100.0 / (10.0 * GRAVITY * 5.0);
        assert!(
            (cot - expected).abs() < 1e-3,
            "CoT={cot}, expected {expected}"
        );
    }

    #[test]
    fn cost_of_transport_zero_distance_is_infinite() {
        let mut t = empty_traj();
        t.forward_displacement = 0.0;
        assert!(CostOfTransport.score(&t).is_infinite());
    }

    #[test]
    fn inverse_cot_handles_infinite() {
        let mut t = empty_traj();
        t.forward_displacement = 0.0;
        assert_eq!(InverseCoT.score(&t), 0.0);
    }

    #[test]
    fn smoothness_reward_monotonic_in_smoothness() {
        let smooth = AgentTrajectory {
            action_smoothness: 0.0,
            ..empty_traj()
        };
        let chattery = AgentTrajectory {
            action_smoothness: 100.0,
            ..empty_traj()
        };
        assert!(SmoothnessReward.score(&smooth) > SmoothnessReward.score(&chattery));
    }

    #[test]
    fn orientation_stability_uses_stddev_not_mean() {
        let steady = AgentTrajectory {
            avg_up_alignment: 0.6,
            up_alignment_stddev: 0.05,
            ..empty_traj()
        };
        let wobbly = AgentTrajectory {
            avg_up_alignment: 0.6,
            up_alignment_stddev: 0.4,
            ..empty_traj()
        };
        assert!(OrientationStability.score(&steady) > OrientationStability.score(&wobbly));
    }

    #[test]
    fn walker_fitness_zero_when_no_displacement() {
        let mut t = empty_traj();
        t.forward_displacement = 0.0;
        t.avg_velocity = 0.0;
        assert_eq!(walker_fitness().score(&t), 0.0);
    }

    #[test]
    fn evaluate_trajectory_descriptor_is_two_dim() {
        let t = empty_traj();
        let (_, objectives, descriptor) = evaluate_trajectory(&t);
        assert_eq!(objectives.len(), 2);
        assert_eq!(descriptor.len(), 2);
        for v in &descriptor {
            assert!((0.0..=1.0).contains(v), "descriptor outside [0,1]: {v}");
        }
    }

    /// A vibrating controller signature (large 2nd derivative, near-zero CoM
    /// bounce, sub-target velocity, no foot transitions) collapses to
    /// near-zero walker fitness, while a walking signature keeps a healthy one.
    #[test]
    fn vibrobot_signature_loses_to_walker_signature() {
        let walker = AgentTrajectory {
            forward_displacement: 1.0,
            avg_velocity: 0.4,
            up_alignment_stddev: 0.1,
            com_bounce: 0.07,
            action_smoothness: 0.3,
            control_jerk: 0.01,
            body_jerk: 8.0,
            foot_transitions_per_sensor: 4.0,
            ..empty_traj()
        };
        let vibrobot = AgentTrajectory {
            forward_displacement: 0.1,
            avg_velocity: 0.01,
            up_alignment_stddev: 0.02,
            com_bounce: 0.001,
            action_smoothness: 0.005,
            control_jerk: 0.5,
            body_jerk: 1.0,
            foot_transitions_per_sensor: 0.0,
            ..empty_traj()
        };
        let f_walker = walker_fitness().score(&walker);
        let f_vibrobot = walker_fitness().score(&vibrobot);
        assert!(
            f_walker > 10.0 * f_vibrobot,
            "walker={f_walker} vibrobot={f_vibrobot}: anti-vibrobot terms not biting"
        );
        assert!(
            f_vibrobot < 0.001,
            "vibrobot fitness {f_vibrobot} should be near zero"
        );
    }

    #[test]
    fn velocity_gate_punishes_slow_motion() {
        let fast = AgentTrajectory {
            avg_velocity: 1.0,
            ..empty_traj()
        };
        let crawl = AgentTrajectory {
            avg_velocity: 0.005,
            ..empty_traj()
        };
        assert_eq!(VelocityGate.score(&fast), 1.0);
        assert!(VelocityGate.score(&crawl) < 0.05);
    }

    #[test]
    fn com_bounce_distinguishes_walking_from_sliding() {
        let bouncing = AgentTrajectory {
            com_bounce: 0.05,
            ..empty_traj()
        };
        let flat = AgentTrajectory {
            com_bounce: 0.0001,
            ..empty_traj()
        };
        assert_eq!(ComBounceReward.score(&bouncing), 1.0);
        assert!(ComBounceReward.score(&flat) < 0.01);
    }

    #[test]
    fn foot_air_time_zero_transitions_zero_score() {
        let lifter = AgentTrajectory {
            foot_transitions_per_sensor: 5.0,
            ..empty_traj()
        };
        let slider = AgentTrajectory {
            foot_transitions_per_sensor: 0.0,
            ..empty_traj()
        };
        assert_eq!(FootAirTimeFactor.score(&lifter), 1.0);
        assert_eq!(FootAirTimeFactor.score(&slider), 0.0);
    }

    /// The second-derivative penalty separates vibration from a same-amplitude
    /// smooth sinusoid by orders of magnitude.
    #[test]
    fn control_jerk_kills_sign_flipping_vibration() {
        // ±0.1 sign-flip vibration → |ΔΔv|=0.4 per tick → squared = 0.16.
        let vibrating = AgentTrajectory {
            control_jerk: 0.16,
            ..empty_traj()
        };
        // Smooth 2 Hz sinusoid, same amplitude → |ΔΔv| ≈ 0.0024 → ≈ 6e-6.
        let smooth = AgentTrajectory {
            control_jerk: 6e-6,
            ..empty_traj()
        };
        let r_vibrating = ControlJerkReward.score(&vibrating);
        let r_smooth = ControlJerkReward.score(&smooth);
        assert!(
            r_smooth > 0.95 && r_vibrating < 0.2,
            "smooth={r_smooth} vibrating={r_vibrating}: ControlJerkReward should differentiate"
        );
    }
}
