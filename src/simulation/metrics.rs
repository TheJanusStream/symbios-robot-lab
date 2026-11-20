//! What each robot's evaluation measures, step by step, and the systems that
//! turn it into fitness: explosion culling, metric accumulation, fitness
//! updates and the optional mid-run perturbation.
//!
//! Poses are read from avian's `Position` and `Rotation`, which are current at
//! every fixed step. A `GlobalTransform` read here (as the lab once did) is
//! refreshed only once per rendered frame, so at 5x speed about ten steps saw
//! the same pose and the bounce and wobble variances came out frame-rate
//! dependent.

use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::brain::{MAX_MOTOR_TORQUE, RobotBrain};
use crate::simulation::clock::GenerationClock;
use crate::simulation::fitness::{
    AgentTrajectory, CostOfTransport, FOOT_TRANSITIONS_TARGET, walker_fitness,
};
use crate::simulation::rng::LabRng;
use crate::simulation::sensors::{AgentTouch, hinge_angle, in_contact};
use crate::simulation::target::WalkTarget;
use crate::simulation::terrain::Ground;
use avian3d::prelude::*;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use rand::Rng;
use symbios_genetics::scorers::Scorer;

/// If any part is farther than this from the body, the robot has exploded.
const EXPLOSION_RADIUS: f32 = 6.0;
/// If any part moves faster than this (m/s), the physics has gone unstable:
/// a working robot's parts stay well under it, a breaking joint's pass it
/// within a few steps.
const MAX_PART_SPEED: f32 = 50.0;
/// A body this far below the ground under it has fallen through.
const MIN_Y: f32 = -5.0;
/// A body this far above the ground under it has been launched into the sky.
const MAX_Y: f32 = 10.0;
/// Recompute each robot's fitness every N steps of the measured window (and
/// always on the generation's last step).
const FITNESS_CALC_INTERVAL: u32 = 4;
/// Where in the measured window the perturbation impulse lands.
const PERTURBATION_POINT: f32 = 0.6;

/// Where and which way the robot was facing when the measured window began.
/// Displacement is projected on that forward direction.
#[derive(Component, Default)]
pub struct FitnessBaseline(pub Option<(Vec3, Vec3)>);

/// Running sums over the measured window, one component per robot (on its
/// body, next to the brain).
#[derive(Component, Default, Debug, Clone)]
pub struct AgentMetrics {
    /// Total robot mass (kg), the denominator of the Cost of Transport.
    pub total_mass: f32,
    /// Touch sensors on the robot (denominator of the support duty factor).
    pub touch_sensor_count: u32,
    /// Coarse commanded-velocity energy `Σ |τ_max · v_target| · dt`, for
    /// the inspector.
    pub energy_consumed: f32,
    /// Mechanical work `Σ |τ_max · Δθ|` from actual joint-angle changes.
    pub mechanical_work: f32,
    pub up_alignment_sum: f32,
    pub up_alignment_sq_sum: f32,
    pub height_sum: f32,
    pub height_sq_sum: f32,
    /// Per driven joint, the commanded velocity one and two ticks ago.
    pub prev_target_velocities: Vec<f32>,
    pub prev_prev_target_velocities: Vec<f32>,
    /// Per hinge, its angle at the previous tick.
    pub prev_joint_angles: Vec<f32>,
    pub action_smoothness_sum: f32,
    pub control_jerk_sum: f32,
    /// Sum over ticks of the mean commanded joint speed.
    pub action_intensity_sum: f32,
    pub prev_body_lin_vel: Vec3,
    pub body_jerk_sum: f32,
    /// Touch sensors in contact, summed over ticks.
    pub touch_contact_ticks: u32,
    /// Contact/air transitions over all touch sensors.
    pub foot_transition_count: u32,
    /// Set once the perturbation impulse has been applied.
    pub perturbed: bool,
    /// Ticks measured so far (denominator of every mean).
    pub metric_ticks: u32,
}

impl AgentMetrics {
    /// Fresh sums for a robot of `total_mass` with `hinges` hinges,
    /// `driven_joints` driven joints and `touch_sensors` touch sensors.
    pub fn new(total_mass: f32, hinges: usize, driven_joints: usize, touch_sensors: usize) -> Self {
        Self {
            total_mass,
            touch_sensor_count: touch_sensors as u32,
            prev_target_velocities: vec![0.0; driven_joints],
            prev_prev_target_velocities: vec![0.0; driven_joints],
            prev_joint_angles: vec![0.0; hinges],
            ..Default::default()
        }
    }

    /// The trajectory these sums describe, for a robot that has moved
    /// `forward_displacement` metres forward in `window_secs` of measurement.
    pub fn trajectory(&self, forward_displacement: f32, window_secs: f32) -> AgentTrajectory {
        let ticks = self.metric_ticks.max(1) as f32;
        let avg_up = (self.up_alignment_sum / ticks).clamp(0.0, 1.0);
        let up_var = (self.up_alignment_sq_sum / ticks - avg_up * avg_up).max(0.0);
        let mean_y = self.height_sum / ticks;
        let y_var = (self.height_sq_sum / ticks - mean_y * mean_y).max(0.0);
        let driven = self.prev_target_velocities.len().max(1) as f32;
        let touches = self.touch_sensor_count;
        AgentTrajectory {
            forward_displacement,
            avg_velocity: if window_secs > f32::EPSILON {
                forward_displacement / window_secs
            } else {
                0.0
            },
            avg_up_alignment: avg_up,
            avg_height: mean_y.max(0.0),
            up_alignment_stddev: up_var.sqrt(),
            com_bounce: y_var.sqrt(),
            mechanical_work: self.mechanical_work,
            mass: self.total_mass.max(f32::EPSILON),
            action_smoothness: self.action_smoothness_sum / (ticks * driven),
            control_jerk: self.control_jerk_sum / (ticks * driven),
            body_jerk: self.body_jerk_sum / ticks,
            support_duty_factor: if touches > 0 {
                self.touch_contact_ticks as f32 / (ticks * touches as f32)
            } else {
                0.0
            },
            // No touch sensors: full credit on the foot term, nothing to satisfy.
            foot_transitions_per_sensor: if touches > 0 {
                self.foot_transition_count as f32 / touches as f32
            } else {
                FOOT_TRANSITIONS_TARGET
            },
            action_intensity: self.action_intensity_sum / ticks,
        }
    }
}

/// Culls robots whose physics blew up, from the first step on, before NaN or
/// infinite values spread through the solver. An exploded agent is marked,
/// despawned, and scores as dominated at the end of the generation.
///
/// Culled robots are genuinely broken, not unlucky. A loaded hinge, usually
/// a long leg's resting on its angle limit, starts vibrating at a few
/// hundred hertz about the axes it should lock; within a few steps its parts
/// move at hundreds of metres per second and the joint tears apart. One or
/// two robots in a generation of 64 do this, more on long-legged bodies and
/// on bodies with a part under half a kilogram. Measured without reliable
/// effect: twice the substeps, weaker motors, softer or no angle limits,
/// joint damping, heavier light parts. Speed caps stop the tearing but leave
/// the limb spinning at the cap, a perpetual motor that carried robots tens
/// of metres, so this cull stays the remedy.
pub fn check_explosion(
    mut commands: Commands,
    brains: Query<(&RobotBrain, &Position)>,
    parts: Query<(&Position, &LinearVelocity), Without<RobotBrain>>,
    mut population: ResMut<Population>,
    ground: Res<Ground>,
    clock: Res<GenerationClock>,
) {
    // Bypass change detection: a changed Population means "respawn" to the
    // spawner, and an exploded robot must stay gone until the next generation.
    let population = population.bypass_change_detection();
    for (index, agent) in population.agents.iter_mut().enumerate() {
        let Some(body) = agent.body_entity else {
            continue;
        };
        let Ok((brain, body_position)) = brains.get(body) else {
            continue;
        };
        let body_pos = body_position.0;
        let height = body_pos.y - ground.height_at(body_pos.x, body_pos.z);
        let exploded = !body_pos.is_finite()
            || height < MIN_Y
            || height > MAX_Y
            || brain.parts.iter().any(|&part| {
                parts.get(part).is_ok_and(|(position, velocity)| {
                    !position.0.is_finite()
                        || velocity.length_squared() > MAX_PART_SPEED * MAX_PART_SPEED
                        || position.0.distance_squared(body_pos)
                            > EXPLOSION_RADIUS * EXPLOSION_RADIUS
                })
            });
        if exploded {
            // Past the generation's end the walk is over and keeps its
            // score; the robot still goes before it can upset the physics.
            if !clock.is_past_end() {
                agent.exploded = true;
                agent.fitness = 0.0;
            }
            agent.body_entity = None;
            if let Some(root) = agent.root_entity.take() {
                commands.entity(root).try_despawn();
            }
            info!("Agent {index} exploded, despawning");
        }
    }
}

/// Accumulates every robot's metrics for one fixed step of the measured
/// window. Runs after `drive_motors`, so it sees this step's commands.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn track_metrics(
    mut brains: Query<(
        Entity,
        &RobotBrain,
        &mut AgentMetrics,
        &Position,
        &Rotation,
        &LinearVelocity,
    )>,
    rotations: Query<&Rotation>,
    hinges: Query<&RevoluteJoint>,
    mut touches: Query<&mut AgentTouch>,
    colliding: Query<&CollidingEntities>,
    clock: Res<GenerationClock>,
    time: Res<Time>,
    ground: Res<Ground>,
) {
    if !clock.is_settled() || clock.is_past_end() {
        return;
    }
    let dt = time.delta_secs();
    let inv_dt = if dt > f32::EPSILON { 1.0 / dt } else { 0.0 };

    // One pass over the touch sensors: contacts (support duty) and
    // contact/air changes (step cycles), per brain.
    let mut contact_counts: HashMap<Entity, u32> = HashMap::default();
    let mut transition_counts: HashMap<Entity, u32> = HashMap::default();
    for mut touch in &mut touches {
        let touching = in_contact(&touch, &colliding);
        if touching {
            *contact_counts.entry(touch.brain_entity).or_default() += 1;
        }
        if touching != touch.prev_in_contact {
            *transition_counts.entry(touch.brain_entity).or_default() += 1;
            touch.prev_in_contact = touching;
        }
    }

    for (brain_entity, brain, mut metrics, position, rotation, lin_vel) in &mut brains {
        let metrics = metrics.as_mut();
        let first_tick = metrics.metric_ticks == 0;

        // The brain's commands: hinges first, then sliders.
        let mut speed_sum = 0.0;
        for (i, v) in brain.commands().enumerate() {
            let v_prev = metrics.prev_target_velocities[i];
            let v_prev2 = metrics.prev_prev_target_velocities[i];
            let dv = v - v_prev;
            metrics.action_smoothness_sum += dv * dv;
            if metrics.metric_ticks > 1 {
                let ddv = v - 2.0 * v_prev + v_prev2;
                metrics.control_jerk_sum += ddv * ddv;
            }
            metrics.prev_prev_target_velocities[i] = v_prev;
            metrics.prev_target_velocities[i] = v;
            speed_sum += v.abs();
        }
        metrics.action_intensity_sum += speed_sum / brain.driven_joint_count().max(1) as f32;

        // The commanded-velocity energy proxy, and mechanical work from the
        // hinges' actual angle changes.
        let hinge_commands = brain.commands().take(brain.hinge_joints.len());
        metrics.energy_consumed +=
            hinge_commands.map(f32::abs).sum::<f32>() * MAX_MOTOR_TORQUE * dt;
        for (i, driven) in brain.hinge_joints.iter().enumerate() {
            let Ok(joint) = hinges.get(driven.entity) else {
                continue;
            };
            let (Ok(rot1), Ok(rot2)) = (rotations.get(joint.body1), rotations.get(joint.body2))
            else {
                continue;
            };
            let angle = hinge_angle(joint, rot1.0, rot2.0);
            if !first_tick {
                let mut delta = angle - metrics.prev_joint_angles[i];
                if delta > std::f32::consts::PI {
                    delta -= std::f32::consts::TAU;
                } else if delta < -std::f32::consts::PI {
                    delta += std::f32::consts::TAU;
                }
                metrics.mechanical_work += MAX_MOTOR_TORQUE * delta.abs();
            }
            metrics.prev_joint_angles[i] = angle;
        }

        if !first_tick {
            let dv_body = lin_vel.0 - metrics.prev_body_lin_vel;
            metrics.body_jerk_sum += dv_body.length_squared() * inv_dt;
        }
        metrics.prev_body_lin_vel = lin_vel.0;

        // The body's local -Y points up at spawn (it is spawned turned over).
        let up = (rotation.0 * Vec3::NEG_Y).dot(Vec3::Y);
        metrics.up_alignment_sum += up;
        metrics.up_alignment_sq_sum += up * up;
        // Height above the ground under the body, so walking over a hill
        // neither lifts the posture score nor counts as bounce.
        let height = position.0.y - ground.height_at(position.0.x, position.0.z);
        metrics.height_sum += height;
        metrics.height_sq_sum += height * height;

        metrics.touch_contact_ticks += contact_counts.get(&brain_entity).copied().unwrap_or(0);
        metrics.foot_transition_count += transition_counts.get(&brain_entity).copied().unwrap_or(0);
        metrics.metric_ticks += 1;
    }
}

/// Updates every robot's fitness and metric fields on its agent. Fitness is
/// 0 while settling; when the measured window opens, the body's pose becomes
/// the baseline displacement is measured from: forward along the way it then
/// faces or, in the target task, the progress it makes toward its target.
pub fn update_fitness_scores(
    mut brains: Query<(
        Entity,
        &RobotBrain,
        &AgentMetrics,
        &Position,
        &Rotation,
        &mut FitnessBaseline,
    )>,
    targets: Query<&WalkTarget>,
    mut population: ResMut<Population>,
    clock: Res<GenerationClock>,
) {
    let population = population.bypass_change_detection();
    if !clock.is_settled() {
        for (_, brain, ..) in &brains {
            if let Some(agent) = population.agents.get_mut(brain.agent_index) {
                agent.fitness = 0.0;
            }
        }
        return;
    }
    let window_step = clock.step() - clock.settle_step();
    if clock.is_past_end()
        || !window_step.is_multiple_of(FITNESS_CALC_INTERVAL) && !clock.is_finished()
    {
        return;
    }

    let window_secs = clock.window_elapsed_secs();
    for (body, brain, metrics, position, rotation, mut baseline) in &mut brains {
        let (base_pos, forward) = *baseline.0.get_or_insert_with(|| {
            let forward = rotation.0 * Vec3::X;
            let forward_xz = Vec3::new(forward.x, 0.0, forward.z);
            let forward = if forward_xz.length_squared() > 1e-6 {
                forward_xz.normalize()
            } else {
                Vec3::X
            };
            (position.0, forward)
        });
        let forward_displacement = match targets.get(body) {
            Ok(target) => target.ground_gained(position.0),
            Err(_) => (position.0 - base_pos).dot(forward).max(0.0),
        };
        let forward_displacement = if forward_displacement.is_finite() {
            forward_displacement
        } else {
            0.0
        };
        let trajectory = metrics.trajectory(forward_displacement, window_secs);
        let fitness = walker_fitness().score(&trajectory);

        let Some(agent) = population.agents.get_mut(brain.agent_index) else {
            continue;
        };
        agent.fitness = if fitness.is_finite() { fitness } else { 0.0 };
        agent.displacement = forward_displacement;
        agent.energy_consumed = metrics.energy_consumed;
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

/// Applies one sideways impulse to each robot, `PERTURBATION_POINT` of the
/// way through its measured window. Off when
/// `LabConfig.perturbation_strength` is 0. The impulse is horizontal and
/// perpendicular to the robot's recorded forward axis (lateral stability is
/// the commonest failure of legged gaits); its sign is random per robot so
/// the population is not pushed one way.
pub fn apply_perturbations(
    mut brains: Query<(&mut AgentMetrics, &FitnessBaseline, Forces)>,
    clock: Res<GenerationClock>,
    config: Res<LabConfig>,
    mut rng: ResMut<LabRng>,
) {
    if config.perturbation_strength <= f32::EPSILON || !clock.is_settled() {
        return;
    }
    let window = clock.total_steps() - clock.settle_step();
    let trigger = clock.settle_step() + (window as f32 * PERTURBATION_POINT) as u32;
    if clock.step() < trigger {
        return;
    }
    for (mut metrics, baseline, mut forces) in &mut brains {
        if metrics.perturbed {
            continue;
        }
        let Some((_, forward)) = baseline.0 else {
            continue;
        };
        let lateral = Vec3::new(-forward.z, 0.0, forward.x).normalize_or_zero();
        let sign = if rng.rng().random_bool(0.5) {
            1.0
        } else {
            -1.0
        };
        forces.apply_linear_impulse(lateral * (sign * config.perturbation_strength));
        metrics.perturbed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_robot_with_no_feet_gets_full_foot_credit() {
        let metrics = AgentMetrics {
            metric_ticks: 10,
            ..AgentMetrics::new(5.0, 2, 2, 0)
        };
        let t = metrics.trajectory(1.0, 2.0);
        assert_eq!(t.foot_transitions_per_sensor, FOOT_TRANSITIONS_TARGET);
        assert_eq!(t.avg_velocity, 0.5);
    }

    #[test]
    fn action_intensity_is_the_window_mean() {
        let metrics = AgentMetrics {
            metric_ticks: 4,
            action_intensity_sum: 8.0,
            ..AgentMetrics::new(5.0, 2, 2, 4)
        };
        assert_eq!(metrics.trajectory(0.0, 1.0).action_intensity, 2.0);
    }

    #[test]
    fn bounce_is_the_standard_deviation_of_height() {
        // Heights 0.4 and 0.6 alternating: mean 0.5, stddev 0.1.
        let metrics = AgentMetrics {
            metric_ticks: 2,
            height_sum: 1.0,
            height_sq_sum: 0.16 + 0.36,
            ..AgentMetrics::new(5.0, 0, 0, 0)
        };
        let t = metrics.trajectory(0.0, 1.0);
        assert!((t.com_bounce - 0.1).abs() < 1e-5, "bounce {}", t.com_bounce);
        assert!((t.avg_height - 0.5).abs() < 1e-6);
    }
}
