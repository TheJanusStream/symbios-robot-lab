//! Fitness scoring for evolved walkers: pure functions of a robot's measured
//! trajectory, composed from `symbios-genetics` scorers.
//!
//! The walker fitness is forward displacement times six factors in `[0, 1]`.
//! Each factor is a gate: a robot that walks passes all six near 1
//! (champions keep about nine tenths of their distance), so among walkers
//! the fitness is very nearly the distance, and a robot that gets about by
//! something the lab does not count as walking fails one of them.
//!
//! | Factor | Fails |
//! |--------|-------|
//! | `OrientationStability` (1 − stddev/0.4) | tipping, wobbling |
//! | `SteadyCommands` (worst joint's command turns, 4 to 8 Hz) | vibrating |
//! | `JerkReward` (1/(1+body_jerk/scale)) | flailing body motion |
//! | `VelocityGate` (clamp v/0.2) | slow sliding |
//! | `FootAirTimeFactor` (transitions/target) | no real step cycles |
//! | `ComBounceReward` (clamp σ_y/0.02) | flat sliding (no bounce) |
//!
//! # Vibration
//!
//! A robot can slide across the floor by shaking its joints, and with no
//! term against it three champions in sixteen did: every command flipped at
//! each evaluation, feet that never left the ground. `SteadyCommands` judges
//! how often a joint's command turns round over the walk, on the joint where
//! it turns most often. As often as a stride on the fastest clock a brain
//! can have (4 Hz) costs nothing; at twice that the factor is 0.
//!
//! What it does not see: a swing of a twentieth of a joint's range or less
//! (see [`TURN_HYSTERESIS`]), which the motors all but swallow; and when in
//! the walk the turns fall, so a burst of shaking within the walk's budget
//! passes, and it is the bounce and step gates that stop a robot sliding on
//! bursts. Pattern-generator brains turn a command once or twice more per
//! stride when a footfall restarts a pattern midway, so theirs meet the gate
//! at slower clocks than 4 Hz.
//!
//! [`TURN_HYSTERESIS`]: crate::simulation::metrics::TURN_HYSTERESIS
//!
//! The lab used to multiply by two factors of another kind, the mean squared
//! step and the mean squared second difference of the commands, averaged
//! over the joints. Brains run every second step, so the two measured the
//! same thing, and it was joint speed, not vibration: a full-range 1 Hz
//! stride on every joint kept an eighth of its distance, and one full-range
//! step, as a foot sensor asks for, cost a leg nearly all of it. That paid
//! for long levers on few, slow joints (champions grew to 2.6 times their
//! starting weight in 100 generations), left a small vibration alone, and
//! an idle joint thinned the average. A turn count has none of that: a
//! large or a fast stride pays nothing, a step is one turn at most, and
//! nothing an added joint does can lower the worst joint's count.
//!
//! # Economy
//!
//! NSGA-II's second objective is how little the joints move for the ground
//! covered: the radians every hinge turned through, each counted as one body
//! size of lever, and the metres every slider slid, per metre walked
//! ([`JointTravel`]). Nothing in it grows with a body's weight or with its
//! size, so neither is a way to score.
//!
//! Three candidates were bred against one another, sixteen runs of 100
//! generations each. This one's champions walked 23.7 m at their starting
//! weight (266 kg). The lab's former estimate of a Cost of Transport, 5000
//! N m for every radian of hinge over weight times distance, called a
//! heavier body more efficient for the same motion: its champions walked
//! 21.9 m and weighed 371 kg. The Cost of Transport from the motors' real
//! work kept bodies light too, and its champions walked a third less far
//! (16.1 m): a joint twitching unloaded costs next to no work, so it does
//! not ask for calm joints, and it is calm joints that walk the same way
//! twice. With no second objective at all (15.2 m), robots twitched, wasted
//! and broke.
//!
//! The work the motors did is still measured (each motor's torque times the
//! angle its hinge turned through, force times travel for a slider, as the
//! physics reports them, in magnitudes: braking a joint costs as driving it
//! does) and shown as the Cost of Transport `E / (m·g·d)`.

use crate::core::genome::MAX_RHYTHM_HZ;
use crate::core::population::Agent;
use crate::simulation::physics::GRAVITY;
use symbios_genetics::scorers::{Multiply, Scorer};

// -- Scoring scales --------------------------------------------------------
//
// These constants encode "how bad is bad" for each gate. A competent walker
// scores about 1 on each factor; what the factor is there to stop scores
// near 0.

/// Stddev(up_alignment) at which `OrientationStability` reaches 0.
const ORIENT_STDDEV_SCALE: f32 = 0.4;
/// How often (Hz) a joint's command may turn round and back at no cost: as
/// often as the fastest clock a brain can read.
pub const STEADY_COMMAND_HZ: f32 = MAX_RHYTHM_HZ;
/// Where `SteadyCommands` reaches 0. The motors are springs of 5 Hz and
/// carry out under a third of a command that turns this often.
pub const VIBRATING_COMMAND_HZ: f32 = 2.0 * STEADY_COMMAND_HZ;
/// Per-tick body jerk² scale; walkers measure 1 to 6 and keep 0.9 or more.
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
/// The joint travel per metre walked at which [`Economy`] is one half.
/// Walkers measure 10 to 30.
const JOINT_TRAVEL_SCALE: f32 = 10.0;
/// The least body size (m) a hinge's travel is counted at, so that a speck
/// of a body does not move its joints for nothing.
const MIN_BODY_SIZE: f32 = 0.25;

/// The number of this fitness formula. A score says nothing beside a score
/// of another formula, so saved champions carry the number of theirs and a
/// robot scored under another arrives without a score (see
/// [`Champion::from_json`]). Raise it with every change to what a walk
/// scores.
///
/// [`Champion::from_json`]: crate::simulation::hall_of_fame::Champion::from_json
pub const SCORING: u32 = 2;

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
    /// How far the hinges turned (rad) and the sliders slid (m), summed
    /// over the driven joints.
    pub hinge_travel: f32,
    pub slider_travel: f32,
    /// The body's size (m): the edge of a cube as roomy as the box around
    /// its parts at rest (see `BodyTraits::size`).
    pub body_size: f32,
    /// Work the motors did (J), in magnitudes: the numerator of CoT.
    pub mechanical_work: f32,
    /// Total robot mass (kg), the denominator of CoT.
    pub mass: f32,
    /// How often the command of the joint that turns most often turned round
    /// and back (Hz): two turns make a cycle.
    pub command_turn_hz: f32,
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

/// The vibration gate: 1 while the command of the joint that turns most
/// often turns no more often than [`STEADY_COMMAND_HZ`], falling to 0 at
/// [`VIBRATING_COMMAND_HZ`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SteadyCommands;
impl Scorer<AgentTrajectory, f32> for SteadyCommands {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        ((VIBRATING_COMMAND_HZ - t.command_turn_hz) / (VIBRATING_COMMAND_HZ - STEADY_COMMAND_HZ))
            .clamp(0.0, 1.0)
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

/// Joint travel per metre walked, lower is better: the radians the hinges
/// turned through, each counted as one body size of lever, and the metres
/// the sliders slid. Infinite when the robot did not progress.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointTravel;
impl Scorer<AgentTrajectory, f32> for JointTravel {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        let distance = t.forward_displacement.max(0.0);
        if distance > f32::EPSILON {
            let travel = t.hinge_travel * t.body_size.max(MIN_BODY_SIZE) + t.slider_travel;
            travel / distance
        } else {
            f32::INFINITY
        }
    }
}

/// [`JointTravel`] as a maximize-me reward in `(0, 1]`, NSGA-II's second
/// objective: `1 / (1 + travel / JOINT_TRAVEL_SCALE)`, 0 for a robot that
/// did not progress.
#[derive(Clone, Copy, Debug, Default)]
pub struct Economy;
impl Scorer<AgentTrajectory, f32> for Economy {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        let travel = JointTravel.score(t);
        if travel.is_finite() {
            1.0 / (1.0 + travel / JOINT_TRAVEL_SCALE)
        } else {
            0.0
        }
    }
}

type WalkerFitness = Multiply<
    Multiply<
        Multiply<
            Multiply<
                Multiply<Multiply<ForwardDisplacement, OrientationStability>, SteadyCommands>,
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
                        Multiply(ForwardDisplacement, OrientationStability),
                        SteadyCommands,
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

/// Objectives [`evaluate_trajectory`] scores: walker fitness and economy.
pub const NUM_OBJECTIVES: usize = 2;

/// `(fitness, objectives, gait)` for a trajectory:
/// - **fitness**: the walker composite;
/// - **objectives**: `[walker_fitness, economy]` for NSGA-II
///   ([`NUM_OBJECTIVES`] of them);
/// - **gait**: `[support_duty_factor, action_intensity_norm]`, the gait
///   traits of the quality-diversity descriptor
///   ([`crate::core::descriptor`]).
pub fn evaluate_trajectory(t: &AgentTrajectory) -> (f32, Vec<f32>, Vec<f32>) {
    let fitness = walker_fitness().score(t);
    let objectives = vec![fitness, Economy.score(t)];
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
        command_turn_hz: agent.command_turn_hz,
        hinge_travel: agent.hinge_travel,
        slider_travel: agent.slider_travel,
        body_size: agent.body_traits.map_or(0.0, |b| b.size),
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
            command_turn_hz: 1.0,
            body_jerk: 5.0,
            hinge_travel: 0.0,
            slider_travel: 0.0,
            body_size: 2.0,
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

    /// Economy is joint travel against distance: half at 10 per metre, 0
    /// for a robot that got nowhere, and better the less the joints move.
    #[test]
    fn economy_is_joint_travel_per_metre_walked() {
        let walk = AgentTrajectory {
            forward_displacement: 10.0,
            hinge_travel: 40.0,
            slider_travel: 20.0,
            body_size: 2.0,
            ..empty_traj()
        };
        // 40 rad at 2 m of lever and 20 m of sliding over 10 m.
        assert_eq!(JointTravel.score(&walk), 10.0);
        assert_eq!(Economy.score(&walk), 0.5);
        let calmer = AgentTrajectory {
            hinge_travel: 15.0,
            ..walk.clone()
        };
        assert!(Economy.score(&calmer) > Economy.score(&walk));
        let stuck = AgentTrajectory {
            forward_displacement: 0.0,
            ..walk.clone()
        };
        assert!(JointTravel.score(&stuck).is_infinite());
        assert_eq!(Economy.score(&stuck), 0.0);
    }

    /// Neither size nor weight is a way to score: the same strides on a
    /// body twice as large cover twice the ground and are as economical, and
    /// a heavier body is no more so.
    #[test]
    fn economy_ignores_size_and_weight() {
        let small = AgentTrajectory {
            forward_displacement: 8.0,
            hinge_travel: 60.0,
            slider_travel: 0.0,
            body_size: 2.0,
            mass: 250.0,
            ..empty_traj()
        };
        let large = AgentTrajectory {
            forward_displacement: 16.0,
            body_size: 4.0,
            mass: 2000.0,
            ..small.clone()
        };
        assert_eq!(Economy.score(&small), Economy.score(&large));
        let heavy = AgentTrajectory {
            mass: 1000.0,
            ..small.clone()
        };
        assert_eq!(Economy.score(&small), Economy.score(&heavy));
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

    /// A walker that passes every gate scores its distance.
    #[test]
    fn among_walkers_the_fitness_is_the_distance() {
        let walker = AgentTrajectory {
            forward_displacement: 12.0,
            avg_velocity: 1.2,
            up_alignment_stddev: 0.0,
            body_jerk: 0.0,
            ..empty_traj()
        };
        assert_eq!(walker_fitness().score(&walker), 12.0);
        let twice_as_far = AgentTrajectory {
            forward_displacement: 24.0,
            ..walker
        };
        assert_eq!(walker_fitness().score(&twice_as_far), 24.0);
    }

    /// A vibrating robot scores nothing however far it slides, and for its
    /// commands alone: this one bounces and lifts its feet as a walker does,
    /// and only its commands, flipping at every evaluation, give it away. A
    /// walker keeps most of its distance.
    #[test]
    fn vibrobot_signature_loses_to_walker_signature() {
        let walker = AgentTrajectory {
            forward_displacement: 10.0,
            avg_velocity: 1.0,
            up_alignment_stddev: 0.02,
            com_bounce: 0.07,
            command_turn_hz: 1.2,
            body_jerk: 3.0,
            foot_transitions_per_sensor: 12.0,
            ..empty_traj()
        };
        let vibrobot = AgentTrajectory {
            forward_displacement: 30.0,
            avg_velocity: 3.0,
            command_turn_hz: 16.0,
            ..walker.clone()
        };
        let f_walker = walker_fitness().score(&walker);
        assert!(f_walker > 8.5, "a walker keeps its distance: {f_walker}");
        assert_eq!(walker_fitness().score(&vibrobot), 0.0);
        // Halfway between steady and vibrating, half the distance is left.
        let shaky = AgentTrajectory {
            command_turn_hz: 6.0,
            ..walker.clone()
        };
        let f_shaky = walker_fitness().score(&shaky);
        assert!((f_shaky - f_walker / 2.0).abs() < 1e-4, "{f_shaky}");
    }

    /// The formula [`SCORING`] numbers: this walk scores exactly this. If a
    /// change to the fitness moves it, raise [`SCORING`], so that champions
    /// saved under the old formula lose their scores.
    #[test]
    fn the_formula_is_the_one_its_number_stands_for() {
        let walk = AgentTrajectory {
            forward_displacement: 12.0,
            avg_velocity: 1.2,
            avg_up_alignment: 0.98,
            avg_height: 0.9,
            up_alignment_stddev: 0.04,
            com_bounce: 0.015,
            mechanical_work: 60_000.0,
            mass: 250.0,
            command_turn_hz: 5.0,
            body_jerk: 12.5,
            hinge_travel: 96.0,
            slider_travel: 12.0,
            body_size: 2.5,
            support_duty_factor: 0.7,
            foot_transitions_per_sensor: 2.4,
            action_intensity: 1.5,
        };
        // 12 m x 0.9 (posture) x 0.75 (commands) x 0.8 (body jerk) x 1
        // (speed) x 0.8 (steps) x 0.75 (bounce).
        let (fitness, objectives, _) = evaluate_trajectory(&walk);
        assert!((fitness - 3.888).abs() < 1e-4, "fitness {fitness}");
        // 96 rad at 2.5 m of lever and 12 m of sliding over 12 m: a joint
        // travel of 21 per metre.
        assert!((objectives[1] - 1.0 / 3.1).abs() < 1e-5, "{objectives:?}");
        // 60 kJ over 250 kg x 9.81 m/s2 x 12 m.
        assert!((CostOfTransport.score(&walk) - 2.0387).abs() < 1e-3);
        assert_eq!(SCORING, 2);
    }

    /// The vibration gate costs a stride at any clock a brain can have
    /// nothing, and leaves nothing of a command that flips at every
    /// evaluation.
    #[test]
    fn steady_commands_pass_every_clock_and_fail_vibration() {
        let at = |hz: f32| {
            SteadyCommands.score(&AgentTrajectory {
                command_turn_hz: hz,
                ..empty_traj()
            })
        };
        assert_eq!(at(0.0), 1.0);
        assert_eq!(at(crate::core::genome::MIN_RHYTHM_HZ), 1.0);
        assert_eq!(at(MAX_RHYTHM_HZ), 1.0);
        assert_eq!(at(6.0), 0.5);
        assert_eq!(at(8.0), 0.0);
        // The fastest a command can turn: every evaluation, 32 times a second.
        assert_eq!(at(16.0), 0.0);
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
}
