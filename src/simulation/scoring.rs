use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::brain::RobotBrain;
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
const EXPLOSION_RADIUS: f32 = 4.0;

/// If any module exceeds this speed (m/s), the robot is physics-unstable.
const MAX_MODULE_SPEED: f32 = 50.0;

const EXPLOSION_RADIUS_SQ: f32 = EXPLOSION_RADIUS * EXPLOSION_RADIUS;
const MAX_MODULE_SPEED_SQ: f32 = MAX_MODULE_SPEED * MAX_MODULE_SPEED;

/// Body below this Y has fallen through the floor.
const MIN_Y: f32 = -5.0;

/// Body above this Y has been launched into the sky.
const MAX_Y: f32 = 20.0;

/// Calculates fitness as horizontal displacement from the settled baseline.
/// During the settling period (first third of a generation), fitness remains 0.
/// Once settling ends, the body's current position is snapshotted as baseline.
/// Robots whose parts have separated (physics explosion) are despawned immediately
/// and receive fitness 0.
///
/// Uses direct entity references from `RobotBrain.module_entities` to check only
/// the owning agent's modules — no global scan or index-based indirection needed.
pub fn update_fitness_scores(
    mut commands: Commands,
    mut brains: Query<(&RobotBrain, &Transform, &mut FitnessBaseline)>,
    modules: Query<(&Transform, &LinearVelocity), Without<RobotBrain>>,
    mut population: ResMut<Population>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
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
        return;
    }

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

        // Snapshot baseline on first frame after settling
        let base = *baseline.0.get_or_insert(body_pos);

        let displacement = body_pos - base;
        let fitness = Vec2::new(displacement.x, displacement.z).length();

        if let Some(agent) = pop.agents.get_mut(idx) {
            agent.fitness = if fitness.is_finite() { fitness } else { 0.0 };
        }
    }
}
