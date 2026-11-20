//! Fitness scoring for evolved walkers.
//!
//! Replaces the legacy `displacement × up_alignment × (height + 0.5)` formula
//! with a shape-agnostic, smoothness-aware composite drawn from the
//! biomechanics and RL-locomotion literature, **plus an anti-"vibrobot"
//! cocktail** added after observing that the previous formula could be
//! reward-hacked by tiny high-frequency joint oscillations sliding across
//! the floor.
//!
//! Quality factors composed multiplicatively into `walker_fitness`:
//!
//! | Factor | Catches | Reference |
//! |--------|---------|-----------|
//! | `OrientationStability` (1 − stddev/0.4) | tipping, wobbling | classic posture term, softened |
//! | `SmoothnessReward` (1/(1+Σ Δv²/N)) | controller chatter | first-derivative penalty |
//! | `ControlJerkReward` (1/(1+Σ ΔΔv²/(N·s))) | **sign-flipping vibration** | second-derivative; ANYmal/Spot RL |
//! | `JerkReward` (1/(1+body_jerk/scale)) | flailing body motion | first-derivative on body velocity |
//! | `VelocityGate` (clamp v/0.2) | **slow vibrobot sliding** | hard minimum-velocity gate |
//! | `FootAirTimeFactor` (transitions/target) | **no real step cycles** | foot air-time reward (RL locomotion) |
//! | `ComBounceReward` (clamp σ_y/0.02) | **flat sliding (no bounce)** | gait-amplitude reward |
//!
//! Energy is estimated from *actual* joint angle change per tick
//! (`Σ |τ_max · Δθ|`), feeding the standard biomechanics
//! Cost-of-Transport metric `E / (m·g·d)`.

use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::brain::{AgentTouch, MAX_MOTOR_TORQUE, RobotBrain};
use crate::simulation::generation::GenerationTimer;
use avian3d::prelude::*;
use bevy::prelude::*;
use rand::Rng;
use symbios_genetics::scorers::{Multiply, Scorer};

/// Baseline position and forward direction for fitness measurement.
/// Starts as `None`; set after the settling period so the initial fall
/// doesn't dominate the fitness signal. Forward direction is captured at
/// baseline time and displacement is projected onto it.
#[derive(Component)]
pub struct FitnessBaseline(pub Option<(Vec3, Vec3)>);

/// Fraction of the generation timer that must elapse before fitness measurement begins.
const SETTLING_FRACTION: f32 = 1.0 / 3.0;

/// If any module is farther than this from the body, the robot has exploded.
const EXPLOSION_RADIUS: f32 = 6.0;

/// If any module exceeds this speed (m/s), the robot is physics-unstable.
const MAX_MODULE_SPEED: f32 = 50.0;

const EXPLOSION_RADIUS_SQ: f32 = EXPLOSION_RADIUS * EXPLOSION_RADIUS;
const MAX_MODULE_SPEED_SQ: f32 = MAX_MODULE_SPEED * MAX_MODULE_SPEED;

/// Body below this Y has fallen through the floor.
const MIN_Y: f32 = -5.0;

/// Body above this Y has been launched into the sky.
const MAX_Y: f32 = 10.0;

/// Only recalculate displacement fitness every N fixed steps.
const FITNESS_CALC_INTERVAL: u32 = 4;

/// Earth gravity (m/s²), used as the `g` term in Cost-of-Transport.
const GRAVITY_MAG: f32 = 9.81;

// -- Scoring scales --------------------------------------------------------
//
// These constants encode "how bad is bad" for each penalty. They were chosen
// so a competent real walker scores ~0.7+ on each factor and a vibrobot
// scores ~0.1- on the targeted factor, while leaving headroom on either side.

/// Stddev(up_alignment) at which `OrientationStability` reaches 0.
/// Real walkers wobble; 0.4 gives them ~0.5+ score for typical motion.
const ORIENT_STDDEV_SCALE: f32 = 0.4;
/// Per-tick-per-joint mean Δv² scale for `SmoothnessReward`. Real walker
/// chatter ~0.2 → reward ~0.83; flailing ~2.0 → reward ~0.33.
const SMOOTHNESS_SCALE: f32 = 1.0;
/// Per-tick-per-joint mean ΔΔv² scale for `ControlJerkReward`. Calibrated
/// so a walker doing a 2 Hz sinusoid scores ~0.8, while a ±ε vibrator
/// (ΔΔv=4ε per tick at any frequency) scores ~0.1 even for tiny ε.
const CONTROL_JERK_SCALE: f32 = 0.025;
/// Per-tick body jerk² scale. avian's units make body jerk land in the
/// 10³–10⁴ range for active robots; scale=50 keeps healthy walkers ~0.5+.
const BODY_JERK_SCALE: f32 = 50.0;
/// Average forward velocity (m/s) at which `VelocityGate` saturates to 1.0.
/// Below this, score ramps linearly to 0. Vibrobots run at ~cm/s.
const VELOCITY_GATE_TARGET: f32 = 0.2;
/// Foot-cycle transitions per touch sensor at which `FootAirTimeFactor`
/// saturates. ~3 cycles in a 10s window = 0.3 Hz step frequency, generous
/// for slow walkers.
const FOOT_TRANSITIONS_TARGET: f32 = 3.0;
/// Stddev(body_y) at which `ComBounceReward` saturates. 2 cm of vertical
/// motion is the floor — anything less is "sliding".
const COM_BOUNCE_TARGET: f32 = 0.02;

/// Lab-specific physics-trajectory state used as the input `S` for our
/// `Scorer<S, f32>` composition. Carries everything `evaluate_trajectory`
/// needs to compute the new composite walker fitness, the secondary CoT
/// objective, and the QD descriptors.
#[derive(Clone, Debug, Default)]
pub struct AgentTrajectory {
    /// Forward displacement projected on the initial forward direction (m).
    pub forward_displacement: f32,
    /// Average forward velocity over the post-settling window (m/s).
    /// Drives `VelocityGate`.
    pub avg_velocity: f32,
    /// Mean of `dot(local_up, world_up)` over the eval window, clamped `[0, 1]`.
    pub avg_up_alignment: f32,
    /// Mean body Y (m) — kept for inspector display, not in fitness.
    pub avg_height: f32,
    /// Standard deviation of `up_alignment` over the eval window.
    /// `OrientationStability` consumes `1 - stddev/scale`.
    pub up_alignment_stddev: f32,
    /// Standard deviation of body Y (m) — the CoM "bounce" amplitude.
    /// Drives `ComBounceReward`.
    pub com_bounce: f32,
    /// Mechanical-work proxy: `Σ |τ_max · Δθ|` from finite-diff joint angles.
    /// Numerator of CoT.
    pub mechanical_work: f32,
    /// Total robot mass (kg). Denominator of CoT (`E / (m·g·d)`).
    pub mass: f32,
    /// Mean per-tick `Σ_joints |Δv_target|²`. Lower = smoother control.
    pub action_smoothness: f32,
    /// Mean per-tick `Σ_joints |a_t − 2·a_{t−1} + a_{t−2}|²`. Lower = less
    /// vibration. Drives `ControlJerkReward`.
    pub control_jerk: f32,
    /// Mean per-tick `|Δv_body|² / Δt`. Lower = smoother body trajectory.
    pub body_jerk: f32,
    /// Mean fraction of touch sensors in contact each tick (`[0, 1]`).
    /// QD axis 0; not in fitness.
    pub support_duty_factor: f32,
    /// Mean per-touch-sensor count of contact↔air transitions over the
    /// window. Drives `FootAirTimeFactor`.
    pub foot_transitions_per_sensor: f32,
    /// Mean per-joint commanded-velocity magnitude. QD axis 1.
    pub action_intensity: f32,
}

// -- Atomic scorers ---------------------------------------------------------

/// Forward displacement (m), the headline distance signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardDisplacement;
impl Scorer<AgentTrajectory, f32> for ForwardDisplacement {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        t.forward_displacement
    }
}

/// Orientation stability: `1 − stddev(up_alignment) / ORIENT_STDDEV_SCALE`,
/// clamped `[0, 1]`. Softened from the original `1 − stddev` so real walkers
/// (which inevitably wobble) aren't crushed by the multiplicative composite.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrientationStability;
impl Scorer<AgentTrajectory, f32> for OrientationStability {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (1.0 - t.up_alignment_stddev / ORIENT_STDDEV_SCALE).clamp(0.0, 1.0)
    }
}

/// First-derivative smoothness: `1 / (1 + action_smoothness/scale)`.
/// Catches sustained chatter (large per-tick Δv); does NOT catch
/// small-amplitude vibration (that needs `ControlJerkReward`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SmoothnessReward;
impl Scorer<AgentTrajectory, f32> for SmoothnessReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.action_smoothness.max(0.0) / SMOOTHNESS_SCALE)
    }
}

/// **Anti-vibrobot core.** Second-derivative penalty:
/// `1 / (1 + control_jerk/scale)`. A ±ε sign-flipping output has |ΔΔv|=4ε
/// per tick regardless of how small ε is — so this term is uniquely
/// sensitive to high-frequency oscillation that a first-derivative penalty
/// can miss. See module docstring.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlJerkReward;
impl Scorer<AgentTrajectory, f32> for ControlJerkReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.control_jerk.max(0.0) / CONTROL_JERK_SCALE)
    }
}

/// Body-jerk reward: `1 / (1 + body_jerk/scale)`. Penalizes high-frequency
/// body acceleration — i.e. flailing limbs jerking the torso around.
#[derive(Clone, Copy, Debug, Default)]
pub struct JerkReward;
impl Scorer<AgentTrajectory, f32> for JerkReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        1.0 / (1.0 + t.body_jerk.max(0.0) / BODY_JERK_SCALE)
    }
}

/// Minimum-velocity gate: `(avg_velocity / target).clamp(0, 1)`. Linear ramp
/// from 0 (stationary) to 1 (≥ target). Vibrobots scoring 1 cm/s are gated
/// to ~5% reward, while real walkers ≥ 20 cm/s get full credit.
#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityGate;
impl Scorer<AgentTrajectory, f32> for VelocityGate {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (t.avg_velocity / VELOCITY_GATE_TARGET).clamp(0.0, 1.0)
    }
}

/// Foot air-time factor: `(transitions_per_sensor / target).clamp(0, 1)`.
/// Counts contact↔air state changes per touch sensor. A vibrobot whose feet
/// never lift scores 0; a real walker cycling feet ~3 times/window gets full
/// credit. Returns 1.0 (no penalty) when the agent has zero touch sensors,
/// so morphologies without feet aren't punished by this term.
#[derive(Clone, Copy, Debug, Default)]
pub struct FootAirTimeFactor;
impl Scorer<AgentTrajectory, f32> for FootAirTimeFactor {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        // Sentinel: foot_transitions_per_sensor is 0 both when the agent
        // genuinely has no transitions AND when it has no touch sensors at
        // all. The build_trajectory caller substitutes target_value when
        // there are no sensors, so this scorer just clamps.
        (t.foot_transitions_per_sensor / FOOT_TRANSITIONS_TARGET).clamp(0.0, 1.0)
    }
}

/// CoM vertical bounce: `(stddev(body_y) / target).clamp(0, 1)`. Walking
/// has characteristic vertical CoM oscillation; flat sliding has none. This
/// is the term that most directly distinguishes walking from vibrobot slip.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComBounceReward;
impl Scorer<AgentTrajectory, f32> for ComBounceReward {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        (t.com_bounce / COM_BOUNCE_TARGET).clamp(0.0, 1.0)
    }
}

/// Cost of Transport (dimensionless): `E / (m · g · d)`. Standard biomechanics
/// efficiency metric. Lower is better. Returns `f32::INFINITY` when distance
/// or mass is zero so non-progressing robots don't earn fake-low CoT scores.
#[derive(Clone, Copy, Debug, Default)]
pub struct CostOfTransport;
impl Scorer<AgentTrajectory, f32> for CostOfTransport {
    fn score(&self, t: &AgentTrajectory) -> f32 {
        let denom = t.mass * GRAVITY_MAG * t.forward_displacement.max(0.0);
        if denom > f32::EPSILON {
            t.mechanical_work / denom
        } else {
            f32::INFINITY
        }
    }
}

/// CoT inverted into a maximize-me reward in `(0, 1]`: `1 / (1 + CoT)`.
/// Used as NSGA-II's second objective so both objectives share a "higher is
/// better" convention.
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

/// Walker fitness composite (anti-vibrobot edition):
///
/// ```text
/// walker_fitness = ForwardDisplacement
///                × OrientationStability   // posture
///                × SmoothnessReward       // 1st-deriv: anti-chatter
///                × ControlJerkReward      // 2nd-deriv: anti-vibration
///                × JerkReward             // body smoothness
///                × VelocityGate           // anti-slow-slide
///                × FootAirTimeFactor      // anti-no-step-cycle
///                × ComBounceReward        // anti-flat-slide
/// ```
///
/// Seven `[0, 1]` quality factors multiplied onto the displacement signal.
/// Vibrobots score near-zero on at least four (ControlJerk, VelocityGate,
/// FootAirTime, ComBounce) so the product collapses; real walkers score
/// ~0.5–0.8 on each.
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

const fn walker_fitness() -> WalkerFitness {
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

/// Computes `(fitness, objectives, descriptor)` for an `AgentTrajectory`.
///
/// - **fitness** — the new shape-agnostic, anti-vibrobot walker composite.
/// - **objectives** — `[walker_fitness, inverse_CoT]` for NSGA-II.
/// - **descriptor** — `[support_duty_factor, action_intensity_norm]` for QD.
pub fn evaluate_trajectory(t: &AgentTrajectory) -> (f32, Vec<f32>, Vec<f32>) {
    let fitness = walker_fitness().score(t);
    let objectives = vec![fitness, InverseCoT.score(t)];

    const ACTION_INTENSITY_NORM: f32 = 5.0;
    let descriptor = vec![
        t.support_duty_factor.clamp(0.0, 1.0),
        (t.action_intensity / ACTION_INTENSITY_NORM).clamp(0.0, 1.0),
    ];
    (fitness, objectives, descriptor)
}

// -- Systems ---------------------------------------------------------------

/// Checks all robots for physics explosion each fixed step. Runs from the
/// first frame (no settling gate) so unstable robots are culled before
/// NaN/Inf values propagate through the physics solver.
pub fn check_explosion(
    mut commands: Commands,
    brains: Query<(&RobotBrain, &Transform)>,
    modules: Query<(&Transform, &LinearVelocity), Without<RobotBrain>>,
    mut population: ResMut<Population>,
) {
    let pop = population.bypass_change_detection();
    if pop.agents.is_empty() {
        return;
    }

    for (idx, agent) in pop.agents.iter_mut().enumerate() {
        let body_entity = match agent.body_entity {
            Some(e) => e,
            None => continue,
        };

        let Ok((brain, transform)) = brains.get(body_entity) else {
            continue;
        };

        let body_pos = transform.translation;
        let mut exploded = !body_pos.is_finite() || body_pos.y < MIN_Y || body_pos.y > MAX_Y;

        if !exploded {
            for &module_entity in &brain.module_entities {
                if let Ok((tf, vel)) = modules.get(module_entity)
                    && (!tf.translation.is_finite()
                        || vel.length_squared() > MAX_MODULE_SPEED_SQ
                        || tf.translation.distance_squared(body_pos) > EXPLOSION_RADIUS_SQ)
                {
                    exploded = true;
                    break;
                }
            }
        }

        if exploded {
            agent.fitness = 0.0;
            agent.body_entity = None;
            if let Some(root) = agent.root_entity.take() {
                commands.entity(root).try_despawn();
            }
            info!("Agent {} exploded, despawning", idx);
        }
    }
}

/// Per-tick metric accumulation. Runs in `FixedUpdate` after `drive_motors`
/// so it sees the freshly-written `target_velocity`. Populates every brain
/// accumulator consumed by `update_fitness_scores` and `evaluate_trajectory`.
///
/// Combines a single `&mut touches` pass with the per-brain loop to update
/// touch contact counts, foot-transition counts, action smoothness,
/// control-jerk, mechanical work, body jerk, posture, and CoM bounce in one
/// `FixedUpdate` step.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn track_metrics(
    mut brains: Query<(Entity, &mut RobotBrain, &GlobalTransform, &LinearVelocity)>,
    rotations: Query<&Rotation>,
    revolute_joints: Query<&RevoluteJoint>,
    prismatic_joints: Query<&PrismaticJoint>,
    mut touches: Query<(&mut AgentTouch, Option<&CollidingEntities>)>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
    time: Res<Time>,
) {
    let settled = gen_timer.active
        && gen_timer.timer.elapsed_secs() >= config.generation_duration * SETTLING_FRACTION;
    if !settled {
        return;
    }

    let dt = time.delta_secs();
    let inv_dt = if dt > f32::EPSILON { 1.0 / dt } else { 0.0 };

    // Single &mut pass over touch sensors:
    //   - count "in contact" → support duty factor
    //   - count contact↔air state changes → foot transitions (step cycles)
    let mut contacts_per_brain: bevy::platform::collections::HashMap<Entity, u32> =
        bevy::platform::collections::HashMap::default();
    let mut transitions_per_brain: bevy::platform::collections::HashMap<Entity, u32> =
        bevy::platform::collections::HashMap::default();
    for (mut touch, colliding) in &mut touches {
        let in_contact = matches!(colliding, Some(c) if !c.is_empty());
        if in_contact {
            *contacts_per_brain.entry(touch.brain_entity).or_insert(0) += 1;
        }
        if in_contact != touch.prev_in_contact {
            *transitions_per_brain.entry(touch.brain_entity).or_insert(0) += 1;
            touch.prev_in_contact = in_contact;
        }
    }

    for (entity, mut brain, body_global, lin_vel) in &mut brains {
        let brain = brain.as_mut();

        // ── Action smoothness (Σ Δv²) AND control-jerk (Σ ΔΔv²) in one pass.
        // ΔΔv requires two prior samples, so the first two tracked ticks
        // accumulate Δv only. Update prev_prev_target_velocities BEFORE
        // overwriting prev_target_velocities so the rolling history stays
        // consistent.
        for (i, &joint_entity) in brain.joint_entities.iter().enumerate() {
            if let Ok(joint) = revolute_joints.get(joint_entity) {
                let v = joint.motor.target_velocity;
                let v_prev = brain.prev_target_velocities[i];
                let v_prev2 = brain.prev_prev_target_velocities[i];
                let dv = v - v_prev;
                brain.action_smoothness_sum += dv * dv;
                if brain.metric_ticks > 1 {
                    let ddv = v - 2.0 * v_prev + v_prev2;
                    brain.control_jerk_sum += ddv * ddv;
                }
                brain.prev_prev_target_velocities[i] = v_prev;
                brain.prev_target_velocities[i] = v;
            }
        }
        let prismatic_offset = brain.joint_entities.len();
        for (i, &joint_entity) in brain.prismatic_entities.iter().enumerate() {
            if let Ok(joint) = prismatic_joints.get(joint_entity) {
                let v = joint.motor.target_velocity;
                let idx = prismatic_offset + i;
                let v_prev = brain.prev_target_velocities[idx];
                let v_prev2 = brain.prev_prev_target_velocities[idx];
                let dv = v - v_prev;
                brain.action_smoothness_sum += dv * dv;
                if brain.metric_ticks > 1 {
                    let ddv = v - 2.0 * v_prev + v_prev2;
                    brain.control_jerk_sum += ddv * ddv;
                }
                brain.prev_prev_target_velocities[idx] = v_prev;
                brain.prev_target_velocities[idx] = v;
            }
        }

        // ── Mechanical work: Σ |τ_max · Δθ| from actual joint angles.
        let mut tick_work = 0.0_f32;
        for (i, &joint_entity) in brain.joint_entities.iter().enumerate() {
            let Ok(joint) = revolute_joints.get(joint_entity) else {
                continue;
            };
            let (Ok(rot1), Ok(rot2)) = (rotations.get(joint.body1), rotations.get(joint.body2))
            else {
                continue;
            };
            let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
            let basis2 = joint.local_basis2().unwrap_or(Quat::IDENTITY);
            let axis = joint.hinge_axis;
            let perp = if axis.x.abs() < 0.9 {
                axis.cross(Vec3::X).normalize()
            } else {
                axis.cross(Vec3::Y).normalize()
            };
            let a1 = (rot1.0 * basis1) * axis;
            let b1 = (rot1.0 * basis1) * perp;
            let b2 = (rot2.0 * basis2) * perp;
            let sin_angle = b1.cross(b2).dot(a1);
            let cos_angle = b1.dot(b2);
            let angle = sin_angle.atan2(cos_angle);

            if brain.metric_ticks > 0 {
                let mut delta = angle - brain.prev_joint_angles[i];
                if delta > std::f32::consts::PI {
                    delta -= std::f32::consts::TAU;
                } else if delta < -std::f32::consts::PI {
                    delta += std::f32::consts::TAU;
                }
                tick_work += MAX_MOTOR_TORQUE * delta.abs();
            }
            brain.prev_joint_angles[i] = angle;
        }
        brain.mechanical_work += tick_work;

        // ── Legacy commanded-velocity energy for inspector display.
        let mut tick_commanded_energy = 0.0_f32;
        for &joint_entity in &brain.joint_entities {
            if let Ok(joint) = revolute_joints.get(joint_entity) {
                tick_commanded_energy += joint.motor.target_velocity.abs() * MAX_MOTOR_TORQUE * dt;
            }
        }
        brain.energy_consumed += tick_commanded_energy;

        // ── Body jerk: |Δv_body|² × inv_dt, integrated over the window.
        if brain.metric_ticks > 0 {
            let dv_body = lin_vel.0 - brain.prev_body_lin_vel;
            brain.body_jerk_sum += dv_body.length_squared() * inv_dt;
        }
        brain.prev_body_lin_vel = lin_vel.0;

        // ── Posture mean+variance, body-Y mean+variance for CoM bounce.
        let (_, rot, translation) = body_global.to_scale_rotation_translation();
        let local_up = rot * Vec3::NEG_Y;
        let up = local_up.dot(Vec3::Y);
        brain.up_alignment_sum += up;
        brain.up_alignment_sq_sum += up * up;
        brain.height_sum += translation.y;
        brain.height_sq_sum += translation.y * translation.y;

        // ── Touch contact + transition counts (precomputed above).
        if let Some(&hits) = contacts_per_brain.get(&entity) {
            brain.touch_contact_ticks += hits;
        }
        if let Some(&t) = transitions_per_brain.get(&entity) {
            brain.foot_transition_count += t;
        }

        brain.metric_ticks += 1;
    }
}

/// Walker fitness using composable upstream `Scorer`s.
///
/// During the settling period (first third of a generation), fitness remains 0.
/// Once settling ends, the body's position is snapshotted as baseline.
/// Explosion detection is handled separately by `check_explosion`.
///
/// Fitness is only recalculated every `FITNESS_CALC_INTERVAL` steps to
/// reduce redundant work.
///
/// Runs in `FixedUpdate`, locked to the physics timestep.
pub fn update_fitness_scores(
    mut brains: Query<(&RobotBrain, &Transform, &mut FitnessBaseline)>,
    mut population: ResMut<Population>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
    mut step_counter: Local<u32>,
) {
    let settled = gen_timer.active
        && gen_timer.timer.elapsed_secs() >= config.generation_duration * SETTLING_FRACTION;

    let pop = population.bypass_change_detection();

    if !settled {
        for (brain, _, _) in &brains {
            if let Some(agent) = pop.agents.get_mut(brain.agent_index) {
                agent.fitness = 0.0;
            }
        }
        *step_counter = 0;
        return;
    }

    *step_counter += 1;
    if !step_counter.is_multiple_of(FITNESS_CALC_INTERVAL) {
        return;
    }

    let elapsed_in_window =
        gen_timer.timer.elapsed_secs() - config.generation_duration * SETTLING_FRACTION;

    for (brain, transform, mut baseline) in &mut brains {
        let idx = brain.agent_index;
        let body_pos = transform.translation;

        let (base_pos, forward) = *baseline.0.get_or_insert_with(|| {
            let fwd_raw = transform.rotation * Vec3::X;
            let fwd_xz = Vec3::new(fwd_raw.x, 0.0, fwd_raw.z);
            let forward = if fwd_xz.length_squared() > 1e-6 {
                fwd_xz.normalize()
            } else {
                Vec3::X
            };
            (body_pos, forward)
        });

        let displacement = body_pos - base_pos;
        let forward_displacement = displacement.dot(forward).max(0.0);

        let trajectory = build_trajectory(brain, forward_displacement, elapsed_in_window);
        let fitness = walker_fitness().score(&trajectory);

        if let Some(agent) = pop.agents.get_mut(idx) {
            agent.fitness = if fitness.is_finite() { fitness } else { 0.0 };
            agent.displacement = if forward_displacement.is_finite() {
                forward_displacement
            } else {
                0.0
            };
            agent.energy_consumed = brain.energy_consumed;
            agent.avg_up_alignment = trajectory.avg_up_alignment;
            agent.avg_height = trajectory.avg_height;
            agent.mechanical_work = trajectory.mechanical_work;
            agent.up_alignment_stddev = trajectory.up_alignment_stddev;
            agent.action_smoothness = trajectory.action_smoothness;
            agent.control_jerk = trajectory.control_jerk;
            agent.body_jerk = trajectory.body_jerk;
            agent.support_duty_factor = trajectory.support_duty_factor;
            agent.foot_transitions_per_sensor = trajectory.foot_transitions_per_sensor;
            agent.com_bounce = trajectory.com_bounce;
            agent.avg_velocity = trajectory.avg_velocity;
            agent.action_intensity = trajectory.action_intensity;
            agent.cost_of_transport = CostOfTransport.score(&trajectory);
        }
    }
}

/// Builds the full `AgentTrajectory` snapshot from a brain's current
/// accumulators. Pulled out so unit tests can call it directly.
pub fn build_trajectory(
    brain: &RobotBrain,
    forward_displacement: f32,
    elapsed_in_window: f32,
) -> AgentTrajectory {
    let ticks = brain.metric_ticks.max(1) as f32;
    let avg_up = (brain.up_alignment_sum / ticks).clamp(0.0, 1.0);
    let mean_up_sq = brain.up_alignment_sq_sum / ticks;
    let up_var = (mean_up_sq - avg_up * avg_up).max(0.0);
    let stddev = up_var.sqrt();
    let avg_height = (brain.height_sum / ticks).max(0.0);
    let mean_y_sq = brain.height_sq_sum / ticks;
    let mean_y = brain.height_sum / ticks;
    let y_var = (mean_y_sq - mean_y * mean_y).max(0.0);
    let com_bounce = y_var.sqrt();

    let driven_joints = (brain.joint_entities.len() + brain.prismatic_entities.len()).max(1) as f32;
    let action_smoothness = brain.action_smoothness_sum / (ticks * driven_joints);
    let control_jerk = brain.control_jerk_sum / (ticks * driven_joints);
    let body_jerk = brain.body_jerk_sum / ticks;
    let support_duty = if brain.touch_sensor_count > 0 {
        brain.touch_contact_ticks as f32 / (ticks * brain.touch_sensor_count as f32)
    } else {
        0.0
    };
    // Sentinel: if there are no touch sensors, the agent gets full credit on
    // FootAirTimeFactor (no requirement to satisfy). Otherwise normalise
    // total transitions per sensor.
    let foot_transitions_per_sensor = if brain.touch_sensor_count > 0 {
        brain.foot_transition_count as f32 / brain.touch_sensor_count as f32
    } else {
        FOOT_TRANSITIONS_TARGET
    };
    let action_intensity = brain
        .prev_target_velocities
        .iter()
        .map(|v| v.abs())
        .sum::<f32>()
        / driven_joints;
    let avg_velocity = if elapsed_in_window > f32::EPSILON {
        forward_displacement / elapsed_in_window
    } else {
        0.0
    };

    AgentTrajectory {
        forward_displacement,
        avg_velocity,
        avg_up_alignment: avg_up,
        avg_height,
        up_alignment_stddev: stddev,
        com_bounce,
        mechanical_work: brain.mechanical_work,
        mass: brain.total_mass.max(f32::EPSILON),
        action_smoothness,
        control_jerk,
        body_jerk,
        support_duty_factor: support_duty,
        foot_transitions_per_sensor,
        action_intensity,
    }
}

/// Applies a one-shot sideways impulse to each robot mid-evaluation. Off when
/// `LabConfig.perturbation_strength == 0`. Each agent is perturbed once per
/// generation, at a deterministic 60% mark of its post-settling window.
///
/// The impulse is in a horizontal direction perpendicular to the agent's
/// recorded forward axis (so it tests lateral stability — the most common
/// failure mode for legged gaits). Direction sign is randomised per-agent to
/// avoid a population-wide bias.
pub fn apply_perturbations(
    mut brains: Query<(&mut RobotBrain, &FitnessBaseline, Forces)>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
) {
    if config.perturbation_strength <= f32::EPSILON || !gen_timer.active {
        return;
    }
    let elapsed = gen_timer.timer.elapsed_secs();
    let trigger_at = config.generation_duration * SETTLING_FRACTION
        + 0.6 * (config.generation_duration * (1.0 - SETTLING_FRACTION));
    if elapsed < trigger_at {
        return;
    }

    let mut rng = rand::rng();
    let strength = config.perturbation_strength;

    for (mut brain, baseline, mut forces) in &mut brains {
        if brain.perturbed {
            continue;
        }
        let Some((_, forward)) = baseline.0 else {
            continue;
        };

        let lateral = Vec3::new(-forward.z, 0.0, forward.x).normalize_or_zero();
        let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let impulse = lateral * (sign * strength);
        forces.apply_linear_impulse(impulse);
        brain.perturbed = true;
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
        let expected = 100.0 / (10.0 * GRAVITY_MAG * 5.0);
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

    /// Anti-vibrobot acceptance test: a vibrating controller signature
    /// (large 2nd-derivative, near-zero CoM bounce, sub-target velocity, no
    /// foot transitions) collapses to near-zero walker_fitness, while an
    /// otherwise-equal walking signature (small ΔΔv, real bounce, real
    /// velocity, real step transitions) keeps a healthy fitness.
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
            avg_velocity: 0.01,               // crawls at ~cm/s
            up_alignment_stddev: 0.02,        // never tilts
            com_bounce: 0.001,                // flat slide
            action_smoothness: 0.005,         // tiny Δv per tick
            control_jerk: 0.5,                // BUT large ΔΔv from sign flips
            body_jerk: 1.0,                   // body barely moves
            foot_transitions_per_sensor: 0.0, // feet never lift
            ..empty_traj()
        };
        let f_walker = walker_fitness().score(&walker);
        let f_vibrobot = walker_fitness().score(&vibrobot);
        assert!(
            f_walker > 10.0 * f_vibrobot,
            "walker={f_walker} vibrobot={f_vibrobot} — anti-vibrobot terms not biting"
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

    /// Mathematical lethality: the second-derivative penalty differentiates
    /// vibration from a same-amplitude smooth sinusoid by orders of magnitude.
    #[test]
    fn control_jerk_kills_sign_flipping_vibration() {
        // Hand-derived: ±0.1 sign-flip vibration → |ΔΔv|=0.4 per tick →
        // squared = 0.16 per tick per joint.
        let vibrating = AgentTrajectory {
            control_jerk: 0.16,
            ..empty_traj()
        };
        // Smooth 2 Hz sinusoid with the same amplitude → |ΔΔv| ≈ 0.0024 →
        // squared ≈ 6e-6 per tick per joint.
        let smooth = AgentTrajectory {
            control_jerk: 6e-6,
            ..empty_traj()
        };
        let r_vibrating = ControlJerkReward.score(&vibrating);
        let r_smooth = ControlJerkReward.score(&smooth);
        assert!(
            r_smooth > 0.95 && r_vibrating < 0.2,
            "smooth={r_smooth} vibrating={r_vibrating} — ControlJerkReward should differentiate"
        );
    }
}
