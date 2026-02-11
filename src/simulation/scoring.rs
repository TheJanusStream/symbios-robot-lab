use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::brain::RobotBrain;
use crate::simulation::generation::GenerationTimer;
use bevy::prelude::*;

/// Baseline position for fitness measurement.
/// Starts as `None`; set to the body's position after the settling period,
/// so the initial fall doesn't dominate the fitness signal.
#[derive(Component)]
pub struct FitnessBaseline(pub Option<Vec3>);

/// Fraction of the generation timer that must elapse before fitness measurement begins.
const SETTLING_FRACTION: f32 = 1.0 / 3.0;

/// Calculates fitness as horizontal displacement from the settled baseline.
/// During the settling period (first third of a generation), fitness remains 0.
/// Once settling ends, the body's current position is snapshotted as baseline.
pub fn update_fitness_scores(
    mut brains: Query<(&RobotBrain, &Transform, &mut FitnessBaseline)>,
    mut population: ResMut<Population>,
    gen_timer: Res<GenerationTimer>,
    config: Res<LabConfig>,
) {
    let settled = gen_timer.active
        && gen_timer.timer.elapsed_secs() >= config.generation_duration * SETTLING_FRACTION;

    for (brain, transform, mut baseline) in &mut brains {
        if !settled {
            // Still in settling period — no fitness yet
            if let Some(agent) = population.agents.get_mut(brain.agent_index) {
                agent.fitness = 0.0;
            }
            continue;
        }

        // Snapshot baseline on first frame after settling
        let base = *baseline.0.get_or_insert(transform.translation);

        let displacement = transform.translation - base;
        let fitness = Vec2::new(displacement.x, displacement.z).length();

        if let Some(agent) = population.agents.get_mut(brain.agent_index) {
            agent.fitness = if fitness.is_finite() { fitness } else { 0.0 };
        }
    }
}
