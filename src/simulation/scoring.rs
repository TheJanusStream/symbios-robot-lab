use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::brain::{MAX_MOTOR_TORQUE, RobotBrain};
use crate::simulation::generation::GenerationTimer;
use avian3d::prelude::*;
use bevy::prelude::*;

/// Baseline position for fitness measurement.
/// Starts as `None`; set to the body's position after the settling period,
/// so the initial fall doesn't dominate the fitness signal.
#[derive(Component)]
pub struct FitnessBaseline(pub Option<Vec3>);

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
/// Explosion checks still run every step for safety.
const FITNESS_CALC_INTERVAL: u32 = 4;

/// Accumulates energy consumed and posture metrics every FixedUpdate tick
/// after the settling period. Energy is estimated as |target_velocity| * max_torque * dt
/// per joint. Posture tracks root up-alignment (dot product with world up) and height.
///
/// Runs in `FixedUpdate` after `drive_motors`, before `update_fitness_scores`.
pub fn track_metrics(
    mut brains: Query<(&mut RobotBrain, &Transform)>,
    joints: Query<&RevoluteJoint>,
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

    for (mut brain, transform) in &mut brains {
        // Energy: sum |target_velocity * max_torque| * dt across all joints
        let mut tick_energy = 0.0_f32;
        for &joint_entity in &brain.joint_entities {
            if let Ok(joint) = joints.get(joint_entity) {
                tick_energy += joint.motor.target_velocity.abs() * MAX_MOTOR_TORQUE * dt;
            }
        }
        brain.energy_consumed += tick_energy;

        // Posture: dot product of local up with world up
        let local_up = transform.rotation * Vec3::NEG_Y;
        brain.up_alignment_sum += local_up.dot(Vec3::Y);
        brain.height_sum += transform.translation.y;
        brain.metric_ticks += 1;
    }
}

/// Walker fitness using displacement, posture, and energy efficiency.
///
/// Primary fitness value: displacement * average_up_alignment.
/// NSGA-II objectives are set in `tick_generation`.
///
/// During the settling period (first third of a generation), fitness remains 0.
/// Once settling ends, the body's position is snapshotted as baseline.
/// Robots whose parts have separated (physics explosion) are despawned immediately
/// and receive fitness 0.
///
/// Explosion checks run every fixed step. Fitness is only
/// recalculated every `FITNESS_CALC_INTERVAL` steps to reduce redundant work.
///
/// Runs in `FixedUpdate`, locked to the physics timestep.
pub fn update_fitness_scores(
    mut commands: Commands,
    mut brains: Query<(&RobotBrain, &Transform, &mut FitnessBaseline)>,
    modules: Query<(&Transform, &LinearVelocity), Without<RobotBrain>>,
    mut population: ResMut<Population>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
    mut step_counter: Local<u32>,
) {
    let settled = gen_timer.active
        && gen_timer.timer.elapsed_secs() >= config.generation_duration * SETTLING_FRACTION;

    // Bypass change detection: fitness updates are internal bookkeeping,
    // not structural changes that spawn_agents needs to react to.
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
    let calc_fitness = step_counter.is_multiple_of(FITNESS_CALC_INTERVAL);

    for (brain, transform, mut baseline) in &mut brains {
        let idx = brain.agent_index;
        let body_pos = transform.translation;

        // Detect physics explosion via body-level signals:
        // 1. NaN/Inf position  2. Out-of-bounds
        let mut exploded = !body_pos.is_finite() || body_pos.y < MIN_Y || body_pos.y > MAX_Y;

        // Check module-level signals via direct entity refs:
        // 3. Module drift  4. Excessive velocity
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
            if let Some(agent) = pop.agents.get_mut(idx) {
                agent.fitness = 0.0;
                if let Some(root) = agent.root_entity.take() {
                    commands.entity(root).try_despawn();
                }
            }
            continue;
        }

        if !calc_fitness {
            continue;
        }

        // Snapshot baseline on first frame after settling
        let base = *baseline.0.get_or_insert(body_pos);

        let displacement = body_pos - base;
        let horiz_displacement = Vec2::new(displacement.x, displacement.z).length();

        // Walker fitness: displacement * average up-alignment
        let avg_up = if brain.metric_ticks > 0 {
            (brain.up_alignment_sum / brain.metric_ticks as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let avg_height = if brain.metric_ticks > 0 {
            (brain.height_sum / brain.metric_ticks as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let fitness = horiz_displacement * avg_up;

        if let Some(agent) = pop.agents.get_mut(idx) {
            agent.fitness = if fitness.is_finite() { fitness } else { 0.0 };
            agent.displacement = if horiz_displacement.is_finite() {
                horiz_displacement
            } else {
                0.0
            };
            agent.energy_consumed = brain.energy_consumed;
            agent.avg_up_alignment = avg_up;
            agent.avg_height = avg_height;
        }
    }
}
