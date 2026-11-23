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
use crate::simulation::brain::{DrivenJoint, MAX_MOTOR_TORQUE, RobotBrain};
use crate::simulation::clock::GenerationClock;
use crate::simulation::fitness::{
    AgentTrajectory, CostOfTransport, FOOT_TRANSITIONS_TARGET, walker_fitness,
};
use crate::simulation::rng::LabRng;
use crate::simulation::sensors::{AgentTouch, hinge_angle, in_contact, slider_extension};
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

/// How far a command must come back from its last extreme to count as
/// having turned round, on the outputs' `[-1, 1]` scale: a twentieth of a
/// joint's range. A swing of this size or less is not a turn, however fast:
/// the motors, springs of 5 Hz, carry out a tenth of a command flipping at
/// every evaluation, so such a ripple moves a joint by less than a
/// two-hundredth of its range.
pub const TURN_HYSTERESIS: f32 = 0.1;

/// Counts how often a value turns round: each time it comes back from the
/// furthest it went by more than [`TURN_HYSTERESIS`]. A stride is two turns
/// whatever its size or shape, a step is one at most, and a command flipping
/// at every evaluation by more than the hysteresis turns every time. The
/// value's first move, away from where it started, is not a turn.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Turns {
    /// The lowest and the highest the value went since it last turned: the
    /// one behind it is where it turned, the one ahead is its extreme so
    /// far. Before its first move, both since the start.
    low: f32,
    high: f32,
    /// Which way it is heading: up (1), down (-1) or nowhere yet (0).
    direction: i8,
    started: bool,
    count: u32,
}

impl Turns {
    /// Takes the value's next sample.
    pub fn feed(&mut self, value: f32) {
        if !self.started {
            self.started = true;
            self.low = value;
            self.high = value;
            return;
        }
        match self.direction {
            0 => {
                self.low = self.low.min(value);
                self.high = self.high.max(value);
                if value - self.low > TURN_HYSTERESIS {
                    self.direction = 1;
                    self.high = value;
                } else if self.high - value > TURN_HYSTERESIS {
                    self.direction = -1;
                    self.low = value;
                }
            }
            1 => {
                if value > self.high {
                    self.high = value;
                } else if self.high - value > TURN_HYSTERESIS {
                    self.count += 1;
                    self.direction = -1;
                    self.low = value;
                }
            }
            _ => {
                if value < self.low {
                    self.low = value;
                } else if value - self.low > TURN_HYSTERESIS {
                    self.count += 1;
                    self.direction = 1;
                    self.high = value;
                }
            }
        }
    }

    /// Turns counted so far.
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Feeds each driven joint's command (hinges, then sliders, as the brain's
/// outputs are ordered) to its turn counter. A broken joint's command moves
/// nothing, so it is not counted while the joint is broken.
pub fn count_command_turns<'a>(
    turns: &mut [Turns],
    joints: impl Iterator<Item = &'a DrivenJoint>,
    outputs: &[f32],
) {
    for ((turns, joint), &output) in turns.iter_mut().zip(joints).zip(outputs) {
        if !joint.broken {
            turns.feed(output);
        }
    }
}

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
    /// Work the motors did (J): each hinge motor's torque times the angle
    /// it turned through, each slider motor's force times its travel, all
    /// taken as magnitudes, so that braking a joint costs as driving it does.
    pub mechanical_work: f32,
    pub up_alignment_sum: f32,
    pub up_alignment_sq_sum: f32,
    pub height_sum: f32,
    pub height_sq_sum: f32,
    /// Per driven joint (hinges, then sliders), the turns of its command.
    pub command_turns: Vec<Turns>,
    /// Per hinge, its angle at the previous tick.
    pub prev_joint_angles: Vec<f32>,
    /// Per slider, its extension at the previous tick.
    pub prev_slider_extensions: Vec<f32>,
    /// How far the hinges turned (rad) and the sliders slid (m), summed
    /// over the driven joints: what the economy objective is made of.
    pub hinge_travel: f32,
    pub slider_travel: f32,
    /// The body's size (m), which a hinge's travel is counted at (see
    /// `BodyTraits::size`). Set at spawn.
    pub body_size: f32,
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
            command_turns: vec![Turns::default(); driven_joints],
            prev_joint_angles: vec![0.0; hinges],
            prev_slider_extensions: vec![0.0; driven_joints.saturating_sub(hinges)],
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
        let touches = self.touch_sensor_count;
        // Two turns make a cycle.
        let most_turns = self.command_turns.iter().map(Turns::count).max();
        let command_turn_hz = if window_secs > f32::EPSILON {
            most_turns.unwrap_or(0) as f32 / (2.0 * window_secs)
        } else {
            0.0
        };
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
            command_turn_hz,
            hinge_travel: self.hinge_travel,
            slider_travel: self.slider_travel,
            body_size: self.body_size,
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
    poses: Query<(&Position, &Rotation)>,
    hinges: Query<&RevoluteJoint>,
    sliders: Query<&PrismaticJoint>,
    motors: Query<&JointForces>,
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
        let driven_joints = brain.driven_joint_count();
        let speed_sum: f32 = brain.commands().map(f32::abs).sum();
        metrics.action_intensity_sum += speed_sum / driven_joints.max(1) as f32;
        count_command_turns(
            &mut metrics.command_turns,
            brain.hinge_joints.iter().chain(&brain.prismatic_joints),
            &brain.outputs,
        );

        // The commanded-velocity energy proxy, and the work the motors did:
        // the torque or force the physics applied for each over the last
        // step (its mean over the substeps, as a magnitude), times how far
        // its joint moved in that step. A broken joint's motor is off and
        // reports none.
        let hinge_commands = brain.commands().take(brain.hinge_joints.len());
        metrics.energy_consumed +=
            hinge_commands.map(f32::abs).sum::<f32>() * MAX_MOTOR_TORQUE * dt;
        for (i, driven) in brain.hinge_joints.iter().enumerate() {
            let Ok(joint) = hinges.get(driven.entity) else {
                continue;
            };
            let (Ok((_, rot1)), Ok((_, rot2))) = (poses.get(joint.body1), poses.get(joint.body2))
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
                if let Ok(motor) = motors.get(driven.entity) {
                    metrics.mechanical_work += motor.motor_force().abs() * delta.abs();
                }
                metrics.hinge_travel += delta.abs();
            }
            metrics.prev_joint_angles[i] = angle;
        }
        for (i, driven) in brain.prismatic_joints.iter().enumerate() {
            let Ok(joint) = sliders.get(driven.entity) else {
                continue;
            };
            let (Ok((p1, r1)), Ok((p2, r2))) = (poses.get(joint.body1), poses.get(joint.body2))
            else {
                continue;
            };
            let extension = slider_extension(joint, (p1.0, r1.0), (p2.0, r2.0));
            if !first_tick && let Ok(motor) = motors.get(driven.entity) {
                let travel = (extension - metrics.prev_slider_extensions[i]).abs();
                metrics.mechanical_work += motor.motor_force().abs() * travel;
                metrics.slider_travel += travel;
            }
            metrics.prev_slider_extensions[i] = extension;
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
        agent.command_turn_hz = trajectory.command_turn_hz;
        agent.hinge_travel = trajectory.hinge_travel;
        agent.slider_travel = trajectory.slider_travel;
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

    /// A bare physics world stepping as the lab's does, one fixed step an
    /// update.
    fn physics_app() -> App {
        use crate::simulation::physics::{FIXED_HZ, GRAVITY, SUBSTEPS};
        use bevy::time::TimeUpdateStrategy;
        use std::time::Duration;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, PhysicsPlugins::default()))
            .insert_resource(SubstepCount(SUBSTEPS))
            .insert_resource(Gravity(Vec3::NEG_Y * GRAVITY))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / FIXED_HZ,
            )));
        app.finish();
        app.cleanup();
        app
    }

    /// A 10 kg rod of 1 m, level, held by one end on a motored hinge: the
    /// rod and its joint.
    fn lever(app: &mut App) -> (Entity, Entity) {
        let world = app.world_mut();
        let base = world.spawn((RigidBody::Static, Position(Vec3::ZERO))).id();
        let rod = world
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(1.0, 0.1, 0.1),
                Mass(10.0),
                Position(Vec3::new(0.5, 0.0, 0.0)),
                // A body at rest falls asleep, and a sleeping joint reports 0.
                SleepingDisabled,
            ))
            .id();
        let mut hinge = RevoluteJoint::new(base, rod)
            .with_local_anchor2(Vec3::new(-0.5, 0.0, 0.0))
            .with_hinge_axis(Vec3::Z);
        hinge.motor = AngularMotor::default();
        let joint = world.spawn((hinge, JointForces::new())).id();
        (rod, joint)
    }

    /// How far above level the rod points (rad).
    fn rise(app: &App, rod: Entity) -> f32 {
        let rotation = app.world().get::<Rotation>(rod).expect("the rod");
        (rotation.0 * Vec3::X).y.asin()
    }

    /// The physics reports the torque a motor exerts, which the motors' work
    /// is summed from: a motor holding a 10 kg rod level by one end carries
    /// the rod's weight at half its length.
    #[test]
    fn a_motor_holding_a_lever_reports_its_load() {
        use crate::simulation::physics::GRAVITY;
        let mut app = physics_app();
        let (rod, joint) = lever(&mut app);
        for _ in 0..192 {
            app.update();
        }
        let torque = app
            .world()
            .get::<JointForces>(joint)
            .expect("joint forces")
            .motor_force();
        let sag = rise(&app, rod);
        let load = 10.0 * GRAVITY * 0.5 * sag.cos();
        assert!(
            (torque - load).abs() < 0.05 * load,
            "motor torque {torque} N m against a load of {load} N m (sag {sag} rad)"
        );
    }

    /// Torque times the angle turned through is work: a motor that raises
    /// the rod slowly does the work of lifting its weight, summed step by
    /// step as the metrics sum it (each step's torque with that step's turn).
    #[test]
    fn the_work_of_a_slow_lift_is_the_weight_times_the_rise() {
        use crate::simulation::physics::GRAVITY;
        let mut app = physics_app();
        let (rod, joint) = lever(&mut app);
        for _ in 0..128 {
            app.update();
        }
        let start = rise(&app, rod);
        let (mut work, mut angle) = (0.0, start);
        // Four seconds up to 0.6 rad, then a second to come to rest.
        for step in 0..320 {
            let target = 0.6 * ((step + 1) as f32 / 256.0).min(1.0);
            app.world_mut()
                .get_mut::<RevoluteJoint>(joint)
                .expect("the hinge")
                .motor
                .target_position = target;
            app.update();
            let now = rise(&app, rod);
            let torque = app
                .world()
                .get::<JointForces>(joint)
                .expect("joint forces")
                .motor_force();
            work += torque.abs() * (now - angle).abs();
            angle = now;
        }
        let lift = 10.0 * GRAVITY * 0.5 * (angle.sin() - start.sin());
        assert!(angle - start > 0.5, "the rod rose {} rad", angle - start);
        assert!(
            (work - lift).abs() < 0.1 * lift,
            "{work} J of motor work for a lift of {lift} J"
        );
    }

    /// A slider's motor reports its force: holding a 10 kg weight up, the
    /// weight.
    #[test]
    fn a_motor_holding_a_weight_on_a_slider_reports_its_load() {
        use crate::simulation::physics::GRAVITY;
        let mut app = physics_app();
        let world = app.world_mut();
        let base = world.spawn((RigidBody::Static, Position(Vec3::ZERO))).id();
        let weight = world
            .spawn((
                RigidBody::Dynamic,
                Collider::cuboid(0.2, 0.2, 0.2),
                Mass(10.0),
                Position(Vec3::ZERO),
                SleepingDisabled,
            ))
            .id();
        let mut slider = PrismaticJoint::new(base, weight).with_slider_axis(Vec3::Y);
        slider.motor = LinearMotor::default();
        let joint = world.spawn((slider, JointForces::new())).id();
        for _ in 0..192 {
            app.update();
        }
        let force = app
            .world()
            .get::<JointForces>(joint)
            .expect("joint forces")
            .motor_force()
            .abs();
        let load = 10.0 * GRAVITY;
        assert!(
            (force - load).abs() < 0.05 * load,
            "motor force {force} N against a weight of {load} N"
        );
    }

    /// A value sampled at the lab's 64 steps a second over `secs` seconds.
    fn turns_of(secs: f32, value: impl Fn(f32) -> f32) -> u32 {
        let mut turns = Turns::default();
        for step in 0..(secs * 64.0) as u32 {
            turns.feed(value(step as f32 / 64.0));
        }
        turns.count()
    }

    /// A stride turns twice a cycle, whatever its size and whether it
    /// swings or steps: ten cycles count twenty turns (less the first, for a
    /// value that starts at an extreme, and the last, if it is not over).
    #[test]
    fn a_stride_turns_twice_a_cycle_whatever_its_size_or_shape() {
        use std::f32::consts::TAU;
        let wide = turns_of(10.0, |t| (TAU * t).sin());
        let narrow = turns_of(10.0, |t| 0.2 * (TAU * t).sin());
        let stepping = turns_of(10.0, |t| (TAU * t).sin().signum());
        for (name, count) in [("wide", wide), ("narrow", narrow), ("stepping", stepping)] {
            assert!((18..=20).contains(&count), "{name}: {count} turns");
        }
    }

    /// A ripple no larger than the hysteresis is not a turn, alone or
    /// riding on a stride.
    #[test]
    fn a_ripple_is_not_a_turn() {
        use std::f32::consts::TAU;
        let ripple = |t: f32| 0.04 * (TAU * 16.0 * t).sin();
        assert_eq!(turns_of(10.0, ripple), 0);
        let riding = turns_of(10.0, |t| (TAU * t).sin() + ripple(t));
        assert!((19..=20).contains(&riding), "{riding} turns");
    }

    /// A command flipping at every evaluation over `swing`, starting at
    /// `start`: brains run every second step, 32 evaluations a second.
    fn flipping(swing: f32, start: f32) -> u32 {
        let mut turns = Turns::default();
        turns.feed(start);
        for evaluation in 0..320 {
            let value = if evaluation % 2 == 0 {
                swing / 2.0
            } else {
                -swing / 2.0
            };
            turns.feed(value);
            turns.feed(value);
        }
        turns.count()
    }

    /// A command that flips at every evaluation by more than the hysteresis
    /// turns at every evaluation, 16 Hz, the vibrator's mark, wherever it
    /// started: at one of its ends, in the middle, or beyond them.
    #[test]
    fn a_flipping_command_turns_at_every_evaluation() {
        for swing in [0.12, 0.18, 0.3, 2.0] {
            for start in [swing / 2.0, 0.0, -swing / 2.0, 0.9, -0.9] {
                let count = flipping(swing, start);
                assert!(
                    (317..=320).contains(&count),
                    "a swing of {swing} from {start}: {count} turns"
                );
            }
        }
        // At the hysteresis itself, and below, nothing is counted.
        assert_eq!(flipping(0.09, 0.0), 0);
        assert_eq!(flipping(0.1, 0.05), 0);
    }

    /// A broken joint's command moves nothing and is not counted; the
    /// others are, each on its own counter.
    #[test]
    fn a_broken_joints_command_is_not_counted() {
        let joint = |broken| DrivenJoint {
            entity: Entity::PLACEHOLDER,
            min: -1.0,
            max: 1.0,
            max_effort: 100.0,
            leg: None,
            broken,
        };
        let joints = [joint(false), joint(true), joint(false)];
        let mut turns = vec![Turns::default(); 3];
        for step in 0..64 {
            let flip = if step % 2 == 0 { 0.5 } else { -0.5 };
            count_command_turns(&mut turns, joints.iter(), &[flip, flip, 0.2]);
        }
        let counts: Vec<u32> = turns.iter().map(Turns::count).collect();
        assert_eq!(counts, [62, 0, 0]);
    }

    /// The joint whose command turns most often sets the frequency; a joint
    /// at rest neither raises nor lowers it.
    #[test]
    fn the_busiest_joint_sets_the_command_turn_frequency() {
        let mut metrics = AgentMetrics {
            metric_ticks: 640,
            ..AgentMetrics::new(5.0, 3, 3, 0)
        };
        for step in 0..640 {
            let t = step as f32 / 64.0;
            metrics.command_turns[0].feed(0.3);
            metrics.command_turns[1].feed((std::f32::consts::TAU * t).sin());
            metrics.command_turns[2].feed((std::f32::consts::TAU * 3.0 * t).sin());
        }
        let hz = metrics.trajectory(1.0, 10.0).command_turn_hz;
        assert!((2.9..=3.0).contains(&hz), "{hz} Hz");
        // No driven joint, no turns.
        let limp = AgentMetrics::new(5.0, 0, 0, 0);
        assert_eq!(limp.trajectory(1.0, 10.0).command_turn_hz, 0.0);
    }

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
